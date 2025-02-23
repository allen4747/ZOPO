import random
import torch
import numpy as np
import copy

import os
cwd = os.getcwd()
os.environ['PATH'] += ':'+cwd

from automatic_prompt_engineer import ape, data
from data.instruction_induction.load_data import load_data, load_init_space
from evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator, exec_evaluator
from automatic_prompt_engineer import evaluate, config, template, data
import os
from misc import get_test_conf, get_conf

import datetime
from misc import set_all_seed, TASKS, tkwargs, N_INIT, N_QUERIES, Logger
from args import parse_args

from optimization import ZOPO
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LMForwardAPI:
    def __init__(self, logger=None, model_name=None, eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, intrinsic_dim=None, n_prompt_tokens=None, few_shot_data=None,
                 HF_cache_dir=None, args=None):

        self.logger=logger

        self.count = 0

        self.n_prompt_tokens = n_prompt_tokens

        # eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate(
            "Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate(
            "Input: [INPUT]\nOutput: [OUTPUT]")

        if args.api_model in ['vicuna', 'mistral']:
            self.api_model = exec_evaluator(args.api_model, self.conf)
        else:
            self.api_model = args.api_model

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_test_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = []
        self.prompts_set = dict()

    def eval_instruct(self, instruction):
        assert isinstance(instruction, list)
        # score of instruction, take in a list of instruction
        print('Instruction: {}'.format(instruction))
        if instruction[0] in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
        else:
            if 'gpt' in self.api_model or 'bison' in self.api_model:
                dev_perf, instruction_score = evaluate.evaluate_prompts(
                    instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation']['method'], self.conf['evaluation'])
                dev_perf = dev_perf.sorted()[1][0]
                self.prompts_set[instruction[0]] = (
                    dev_perf, instruction_score)
            else:
                outputs = self.api_model.evaluate(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data,
                                        self.conf['evaluation']).sorted()
                dev_perf = outputs[1][0]
                instruction_score = dev_perf 
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)

        if dev_perf > self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_instruction = [instruction]
        elif dev_perf == self.best_dev_perf:
            self.best_instruction.append(instruction)

        self.num_call += 1
        print('STEPS:[{}]. Dev perf: {}. Best dev perf: {}'.format(
            self.num_call,
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        # if (self.num_call>=40) and (self.num_call % 20 == 0):
        if (self.num_call<N_QUERIES+N_INIT) and (self.num_call % 10 == 0):
            if self.best_instruction == []:
                self.logger.record(self.num_call, "", self.test_perf, self.best_dev_perf)
            else:
                self.test_perf=self.run_test(self.best_instruction[-1])
                self.logger.record(self.num_call, self.best_instruction[-1], self.test_perf, self.best_dev_perf)
        return dev_perf, instruction_score
    
    def run_test(self, best_prompt):
        test_res = ape.evaluate_prompts(prompts=best_prompt,
                                        eval_template=self._eval_template,
                                        eval_data=self._test_data,
                                        few_shot_data=self._prompt_gen_data,
                                        demos_template=self._demos_template,
                                        conf=self._test_conf,
                                        base_conf=self._base_conf)
        test_res = test_res[0]
        best_score = test_res.sorted()[1][0]
        return best_score
    
    def prepare_test(self, test_conf, eval_template, test_data, prompt_gen_data, demos_template, base_conf):
        self._test_conf = test_conf
        self._eval_template = eval_template
        self._test_data = test_data
        self._prompt_gen_data = prompt_gen_data
        self._demos_template = demos_template
        self._base_conf = base_conf


    def return_best_prompt(self):
        # return self.best_instruction
        return self.best_instruction[-1]

    def return_prompts_set(self):
        return self.prompts_set


def run(args):
    task, HF_cache_dir = args.task, args.HF_cache_dir
    intrinsic_dim, n_prompt_tokens = args.intrinsic_dim, args.n_prompt_tokens
    query_dir = args.query_dir

    assert (args.task in TASKS), 'Task not found!'

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

    conf = get_conf(task, eval_data, gpt_model=args.api_model)
    test_conf = get_test_conf(task, test_data, gpt_model=args.api_model)

    # make the demo automatically
    subsampled_data = data.subsample_data(
        prompt_gen_data, conf['generation']['num_demos'])
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]



    try:
        instruct_emb_pairs = load_init_space(args.task, query_dir)
        ## Only for ChatGPT
        # for key, value in instruct_emb_pairs.items():
        #     instruct_emb_pairs[key] = ['The instruction is to ' + value[0].strip()]
    except:
        print(args.task, ' should be genereated first!')
        breakpoint()
    print('Total instruction candidates: ', len(instruct_emb_pairs))

    set_all_seed(0)
    embeddings = random.sample(
        list(instruct_emb_pairs.keys()), N_INIT)  # list of prompts
    embeddings = [torch.tensor(embed) for embed in embeddings]
    assert len(embeddings[0].size()) == 1
    input_dim = embeddings[0].size(0)
    # Init the optimization model
    zopo_opts = {
        'maxiter': N_INIT + N_QUERIES,
        'lr': args.lr,
        'max_steps': 2,
        'input_dim': input_dim,
        'tolerance': args.tolerance,
        'neighbors': args.neighbors,
        'nn_depth': args.nn_depth,
        'nn_width': args.nn_width,
        'gp_queries': args.gp_queries,
        'uncertainty_count': args.uncertainty_count,
        'uncertainty_thred': args.uncertainty_thred
    }
    model = ZOPO(zopo_opts)

    logger = Logger('logs',
                    "-".join(str(datetime.datetime.now()).split(' ')+[task]),
                    ('seed', args.seed),
                    ('embedding', 'sentence_emb'),
                    ('dataset', task),
                    ('api_model', args.api_model),
                    ('query_dir', query_dir),
                    ('intrinsic_dim', intrinsic_dim),
                    ('n_token', n_prompt_tokens),
                    ('lr', zopo_opts['lr']),
                    ('inititer', N_INIT),
                    ('maxiter', zopo_opts['maxiter']),
                    ('max_steps', zopo_opts['max_steps']),
                    ('tolerance', zopo_opts['tolerance']),
                    ('neighbors', zopo_opts['neighbors']),
                    ('nn_depth', zopo_opts['nn_depth']),
                    ('nn_width', zopo_opts['nn_width']),
                    ('gp_queries', zopo_opts['gp_queries']),
                    ('uncertainty_count', zopo_opts['uncertainty_count']),
                    ('uncertainty_thred', zopo_opts['uncertainty_thred']),
                    ('algo', "zopo"),
                    )

    model_forward_api = LMForwardAPI(logger= logger, model_name=args.model_name, eval_data=eval_data, init_prompt=init_prompt,
                                     init_qa=init_qa, conf=conf, base_conf=base_conf, prompt_gen_data=prompt_gen_data,
                                     intrinsic_dim=intrinsic_dim, n_prompt_tokens=n_prompt_tokens, HF_cache_dir=HF_cache_dir, args=args)
    model_forward_api.prepare_test(test_conf,
                                    eval_template,
                                    test_data,
                                    prompt_gen_data,
                                    demos_template,
                                    base_conf)
    model.api = model_forward_api

    init_emb_queries = []


    print("Initialization")
    with torch.no_grad():
        for emb in embeddings:
            emb_tuple = tuple(emb.cpu().numpy().tolist())
            instruct = instruct_emb_pairs[emb_tuple]
            dev_score = model_forward_api.eval_instruct(instruct)
            init_emb_queries += [(emb.cpu().numpy(), dev_score[0])]

    # Query with pairs
    model.init_query(init_emb_queries)
    while not model.stop():
        solutions = model.ask(instruct_emb_pairs)
        with torch.no_grad():
            outputs = model_forward_api.eval_instruct(solutions[0])
        model.tell(solutions[1], outputs[0])
        if outputs[0] == 1.0:
            break

    # Test
    print('Evaluate on test data...')
    best_prompt = model_forward_api.return_best_prompt()
    improved_count = model_forward_api.count
    print("Best instruction:")
    print(best_prompt)

    # Evaluate on test data
    print('Evaluating on test data...')

    if 'gpt' in args.api_model or 'bison' in args.api_model:
        best_score = model_forward_api.run_test(best_prompt)
    else:
        conf = config.update_config(test_conf, base_conf)
        eval_template = template.EvalTemplate(eval_template)
        demos_template = template.DemosTemplate(demos_template)
        outputs = model_forward_api.api_model.evaluate(best_prompt, 
                                                        eval_template, 
                                                        test_data,
                                                        demos_template,
                                                        prompt_gen_data,
                                                        conf['evaluation']).sorted()
        best_score = outputs[1][0]


    print("Best instruction is:")
    print(best_prompt)
    logger.save(best_prompt, best_score, model_forward_api.best_dev_perf)
    return best_score, improved_count, best_prompt


if __name__ == '__main__':
    args = parse_args()
    # evaluation budget
    print(
        f"Using a total of {N_INIT + N_QUERIES} function evaluations")
    set_all_seed(args.seed)
    test_score, improved_count, prompts = run(args=args)
    print("Finished!!!")
    print(f'Test score on ChatGPT: {test_score}')