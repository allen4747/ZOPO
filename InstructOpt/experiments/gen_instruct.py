import random
import torch
import numpy as np
import copy

import os
cwd = os.getcwd()
os.environ['PATH'] += ':'+cwd

from automatic_prompt_engineer import ape, data
from experiments.data.instruction_induction.load_data import load_data, load_query_data, save_query_data, load_init_space, save_init_space
from experiments.data.instruction_induction.load_data import load_data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator, exec_evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from automatic_prompt_engineer import evaluate, config, template, data
import os
import re
from misc import get_test_conf, get_conf

from torch.quasirandom import SobolEngine
import time
import argparse
from misc import set_all_seed, TASKS, tkwargs
from args import parse_args

from optimization import ZORD
from misc import N_TOKENS
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LMForwardAPI:
    def __init__(self, model_name=None, eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, intrinsic_dim=None, n_prompt_tokens=None, few_shot_data=None,
                 HF_cache_dir=None, args=None):

        kwargs = {
            'torch_dtype': torch.float16,
            'use_cache': True
            }
        if model_name in ["vicuna", "wizardlm", 'openchat']:
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_cache_dir, low_cpu_mem_usage=True, **kwargs
            ).cuda()

            self.tokenizer = AutoTokenizer.from_pretrained(
                HF_cache_dir,
                model_max_length=1024,
            )
        else:
            raise NotImplementedError

        self.count = 0
        if model_name in ['vicuna', 'wizardlm']:
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['USER:', 'ASSISTANT:']
        elif model_name == 'alpaca':
            self.system_prompt= "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.role = ["### Instruction:", "### Response:"]
        else:
            NotImplementedError

        self.init_token = init_prompt[0] + init_qa[0]
        input_text = f"{self.system_prompt} USER:{self.init_token} ASSISTANT:"
        if model_name in ['mistral', 'vicuna', 'wizardlm']:
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()
            self.example_embed = self.embedding[input_ids]
            self.start_embed = self.embedding[self.tokenizer('\n', return_tensors="pt").input_ids.cuda()]
        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens

        # eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate(
            "Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate(
            "Input: [INPUT]\nOutput: [OUTPUT]")

        if args.api_model in ['llama', 'flan-t5']:
            self.api_model = exec_evaluator(args.api_model, self.conf)
        else:
            self.api_model = args.api_model
        

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = []
        self.prompts_set = dict()

    def init_eval(self, prompt_embedding):
        # when init, prompt -> embeddings, instruction
        if isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(
                prompt_embedding).type(torch.float32)  # z
            prompt_embedding = prompt_embedding.reshape(
                1, self.n_prompt_tokens, -1)
        elif isinstance(prompt_embedding, torch.Tensor):
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = prompt_embedding.reshape(
                1, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )

        prompt_embedding = prompt_embedding.to(
            device=self.example_embed.device, dtype=self.example_embed.dtype)
        # input_embed = torch.cat((self.start_embed, prompt_embedding, self.example_embed), 1)
        input_embed = torch.cat((prompt_embedding, self.example_embed), 1)

        outputs = self.model.generate(
            inputs_embeds=input_embed, 
            return_dict_in_generate=True, 
            output_hidden_states=True, 
            max_new_tokens=64,
            temperature=0.0, 
            do_sample=False)

        # Get embeddings at all token position
        # idx_embed = len(outputs[0][0].cpu().numpy().tolist())
        # output_embeddings = []
        # for i in range(idx_embed-1):
        #     output_embeddings.append(
        #         outputs[1][i][-1][0, -1].detach().cpu().clone())
        # Get embeddings at the first token position
        output_embeddings = [outputs.hidden_states[0][-1][0, -1]]

        output_ids = outputs[0]
        instruction = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        # postprocess instruction
        instruction[0] = '.'.join(instruction[0].split('.')[:-1]+[''])
        return instruction, output_embeddings


    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set


def run(args):
    task, HF_cache_dir = args.task, args.HF_cache_dir
    intrinsic_dim, n_prompt_tokens = args.intrinsic_dim, args.n_prompt_tokens

    query_dir = args.query_dir

    assert args.task in TASKS, 'Task not found!'

    induce_data, test_data = load_data('induce', task), load_data('eval', task)

    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data, eval_data = data.create_split(
        induce_data, prompt_gen_size)

    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                           for output in prompt_gen_data[1]]

    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    # change the evaluation template
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOutput: [OUTPUT]"
    init_prompt = ['\n']
    prompt_gen_template = "I gave a friend a instruction and several inputs. Based on the instruction she produced " \
                          "the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to?"

    base_conf = '../experiments/configs/instruction_induction.yaml'
    conf = get_conf(task, eval_data)

    # make the demo automatically
    subsampled_data = data.subsample_data(
        prompt_gen_data, conf['generation']['num_demos'])
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]
    
    model_forward_api = LMForwardAPI(model_name=args.model_name, eval_data=eval_data, init_prompt=init_prompt,
                                     init_qa=init_qa, conf=conf, base_conf=base_conf, prompt_gen_data=prompt_gen_data,
                                     intrinsic_dim=intrinsic_dim, n_prompt_tokens=n_prompt_tokens, HF_cache_dir=HF_cache_dir, args=args)


    # dict_size = 10000
    dict_size = 5000
    # instruct_size = 2000
    instruct_size = 500
    count_instruct = 0
    
    instruct_emb_pairs = {}
    instruction_dict = {}
    inits_prompts = None

    # Initialization and Save
    inits_prompts = SobolEngine(dimension=intrinsic_dim,
                                scramble=True, seed=0).draw(dict_size)
    with torch.no_grad():
        linear_fn = torch.nn.Linear(
            intrinsic_dim, n_prompt_tokens * 5120, bias=False).cuda()
        set_all_seed(args.seed)
        for p in linear_fn.parameters():
            torch.nn.init.uniform_(p, -1, 1)
        inits_prompts = linear_fn(inits_prompts.cuda()).cpu()
    inits_prompts=inits_prompts[torch.randperm(inits_prompts.size()[0])]
    for i, prompt in enumerate(inits_prompts):
        instruct, embedding = model_forward_api.init_eval(prompt)
        emb_tuple = tuple(embedding[0].cpu().numpy().tolist())
        if (instruct[0] not in instruction_dict) and ((instruct[0].startswith('The instruction')) or (instruct[0].startswith('It appears that the instruction')) or (instruct[0].startswith('It seems like the instruction'))):
            if instruct[0].startswith('It appears that the instruction'):
                instruct[0] = instruct[0].replace('It appears that the instruction', 'The instruction')
            elif instruct[0].startswith('It seems like the instruction'):
                instruct[0] = instruct[0].replace('It seems like the instruction', 'The instruction')
            print('Steps: [{}]: {}'.format(count_instruct, instruct))
            count_instruct += 1
            instruction_dict[instruct[0]] = 1
            instruct_emb_pairs[emb_tuple] = instruct
        else:
            print('Omitted Generation!')
            print(instruct[0])
        if count_instruct >= instruct_size:
            break

    save_init_space(args.task, instruct_emb_pairs, query_dir)

    return None


if __name__ == '__main__':
    args = parse_args()
    # evaluation budget
    set_all_seed(args.seed)
    run(args=args)
    print("Generation Finished!!!")
