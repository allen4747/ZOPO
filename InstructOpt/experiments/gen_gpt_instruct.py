from misc import N_TOKENS
from optimization import ZORD
from args import parse_args
from misc import set_all_seed, TASKS, tkwargs
from torch.quasirandom import SobolEngine
from misc import get_test_conf, get_conf
from sentence_transformers import SentenceTransformer
from automatic_prompt_engineer import evaluate, config, template, data
import experiments.evaluation.instruction_induction.gpt_query as gpt_query
import experiments.evaluation.instruction_induction.evo_query as evo_query
import experiments.data.instruction_induction.generate_emb as generate_emb
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from experiments.data.instruction_induction.load_data import load_data
from experiments.data.instruction_induction.load_data import load_data, save_prompts, load_prompts
from automatic_prompt_engineer import ape, data
import random
import torch
import numpy as np
import copy

import os
cwd = os.getcwd()
os.environ['PATH'] += ':'+cwd


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    task = args.task
    intrinsic_dim, n_prompt_tokens = args.intrinsic_dim, args.n_prompt_tokens
    query_dir = args.query_dir

    assert args.task in TASKS, 'Task not found!'

    induce_data, test_data = load_data('induce', task), load_data('eval', task)

    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    # Induce data is split into prompt_gen_data and eval_data
    set_all_seed(args.seed)
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
    # prompt_gen_template = "I gave my friend a detailed instruction and several inputs. The wrote an output for every one of the inputs based on my instruction. Here are the input-output pairs:[full_DEMO]\n\nThe detailed instruction I gave was to? "
    # prompt_gen_template = "I gave my friend a specific instruction, along with several inputs. For each input, my friend generated a output based on the instruction. Here are the input-output pairs that demonstrate how my friend interpreted and applied the instruction: [full_DEMO]\n\nTo ensure precision in the responses, the instruction was designed to be comprehensive and unambiguous. The detailed instruction I gave was: "
    prompt_gen_template = "I gave a friend a instruction. Based on the instruction they produced " \
                          "the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"

    base_conf = '../experiments/configs/instruction_induction.yaml'
    gpt_model = 'gpt-3.5-turbo'
    conf = {
        'generation': {
            'num_subsamples': 100,
            # 'num_subsamples': 2,
            'num_demos': 5,
            'num_prompts_per_subsample': 50,
            # 'num_prompts_per_subsample': 5,
            'model': {
                'gpt_config': {
                    'model': gpt_model,
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            'model': {
                'gpt_config': {
                    'model': gpt_model,
                }
            }
        },
        'whitebox': {
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            'num_few_shot': 5,
            'model': {
                'name': 'Vicuna_Forward',
                'batch_size': 20,
                'gpt_config': {
                    'model': gpt_model,
                }
            }
        }
    }
    conf = config.update_config(conf, base_conf)

    init_prompts = ape.get_gpt_prompts(eval_template=eval_template,
                                    prompt_gen_data=prompt_gen_data,
                                    eval_data=eval_data,
                                    conf=conf,
                                    base_conf=base_conf,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)
    unique_instructions = set()
    instruction_dict = {}


    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # for i, instruct in enumerate(init_prompts):
    #     print('Step: {} Original Instruction: {}'.format(i, instruct))
    #     try:
    #         emb = model.encode([instruct])[0]
    #         emb = emb.astype('float16')
    #     except:
    #         breakpoint()
    #     # emb = generate_emb.instruct_to_emb([instruct])
    #     instruction_dict[tuple(emb)] = [instruct]

    batch_size = 50
    for i in range(0, len(init_prompts), batch_size):
        print('Step: {} '.format(i))
        prompts = init_prompts[i:i+batch_size]
        embs = model.encode(prompts)
        for emb, instruct in zip(embs, prompts):
            emb = emb.astype('float16')
            instruction_dict[tuple(emb)] = [instruct] 

    mutate_instruction_dict = dict(instruction_dict)

    save_prompts(args.task, query_dir, mutate_instruction_dict)



if __name__ == '__main__':
    args = parse_args()
    # evaluation budget
    set_all_seed(args.seed)
    run(args=args)
    print("Generation Finished!!!")
