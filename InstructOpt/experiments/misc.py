import torch
import random
import numpy as np
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
import os
import errno
import sys
import json

TASKS = [
    'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
    'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation',
    'num_to_verbal', 'active_to_passive', 'singular_to_plural', 'rhymes',
    'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
    'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
    'translation_en-fr', 'word_in_context', 'auto_categorization', 'auto_debugging', 'ascii', 'cs_algorithms',
    'periodic_elements', 'word_sorting', 'word_unscrambling', 'odd_one_out', 'object_counting', 'ag_news', 'samsum',
    'sst2', 'sst5', 'subj', 'trec', 'cola', 'mrpc',  'qqp', 'mnli_matched', 'mnli_mismatched', 'qnli', 'rte'
]


SMOKE_TEST = os.environ.get("SMOKE_TEST")
# bayesian opt
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

N_INIT = 40
N_QUERIES = 165 - N_INIT

N_TOKENS = 1


def get_test_conf(task, test_data, gpt_model='gpt-3.5-turbo-0301'):
    test_conf = {
        'generation': {
            'num_subsamples': 3,
            'num_demos': 5,
            'num_prompts_per_subsample': 0,
            'model': {
                'gpt_config': {
                    'model': gpt_model,
                }
            }
        },
        'evaluation': {
            # option: accuracy (cannot use likelihood here due to the textual outputs from ChatGPT do not have log prob)
            'method': exec_accuracy_evaluator,
            'num_samples': min(100, len(test_data[0])),
            'task': task,
            'model': {
                "name": "GPT_forward",
                        'gpt_config': {
                            'model': gpt_model,
                        }
            }
        }
    }
    return test_conf


def get_conf(task, eval_data, gpt_model='gpt-3.5-turbo-0301'):
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 5,
            'num_prompts_per_subsample': 20,
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
    return conf


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    """
    Base Logger object

    Initializes the log directory and creates log files given by name in arguments.
    Can be used to append future log values to each file.
    """

    def __init__(self, log_folder, exp_name, *args):
        self.log_dir = os.path.join(log_folder, exp_name + '.json')

        try:
            os.makedirs(log_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        init_setting = {}
        for (name, value) in args:
            init_setting[name] = value
        self.data = {'init_setting': init_setting}

        self.instructions = []

        self.trajectory = []

    def log_instruct(self, instruction, score):
        self.instructions.append((instruction, score))
    
    def record(self, t, best_instruct, test_score, dev_score):
        self.trajectory.append((t, best_instruct, test_score, dev_score))

    def save(self, best_instruct, test_score, dev_score):
        self.data['dev_score'] = dev_score
        self.data['test_score'] = test_score
        self.data['best_instruct'] = best_instruct
        self.data['instructions'] = self.instructions

        self.data['trajectory'] = self.trajectory

        with open(self.log_dir, 'w') as f:
            json.dump(self.data, f)
