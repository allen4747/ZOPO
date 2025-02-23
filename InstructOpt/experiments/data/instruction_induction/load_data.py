import os
import json
import random
import pickle

induce_data_path = os.path.join(os.path.dirname(__file__), 'raw/induce/')
eval_data_path = os.path.join(os.path.dirname(__file__), 'raw/execute/')

nlp_induce_data_path = os.path.join(os.path.dirname(__file__), 'nlptasks/induce/')
nlp_eval_data_path = os.path.join(os.path.dirname(__file__), 'nlptasks/execute/')

root_path = os.path.dirname(__file__)

# Get a list of tasks (by looking at the names of the files in the induced directory)
tasks = [f.split('.')[0] for f in os.listdir(induce_data_path)]

nlp_tasks = ['ag_news', 'samsum', 'sst2', 'sst5', 'subj', 'trec', 'cola', 'mrpc',  'qqp', 'mnli_matched', 'mnli_mismatched', 'qnli', 'rte']

def load_data(type, task):
    if task in nlp_tasks:
        base_path = nlp_induce_data_path if type == 'induce' else nlp_eval_data_path
    else:
        base_path = induce_data_path if type == 'induce' else eval_data_path
    path = base_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    num_examples = len(examples)

    inputs, outputs = [], []
    for i in range(num_examples):
        data = examples[str(i + 1)]
        if task == 'cause_and_effect':
            cause, effect = data['cause'], data['effect']
            # Pick an order randomly
            if random.random() < 0.5:
                input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
            else:
                input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
            output_ = [cause]
        elif task == 'common_concept':
            items = data['items']
            # Make comma separated list of items
            input_ = ', '.join(items[:-1])
            output_ = data['all_common_concepts']
        elif task == 'rhymes':
            input_, output_ = data['input'], data['other_rhymes']
        elif 'translation' in task:
            input_, output_ = data['input'], data['possible_translations']
        else:
            input_, output_ = data['input'], [data['output']]
        inputs.append(input_)
        outputs.append(output_)
    return inputs, outputs


def load_query_data(task, query_dir):
    query_data_path = os.path.join(root_path, query_dir)
    file = open(query_data_path + task + "_init_emb_queries.pkl", 'rb')
    init_emb_queries = pickle.load(file)
    file.close()
    file = open(query_data_path + task + "_prompts_set.pkl", 'rb')
    prompts_set = pickle.load(file)
    file.close()
    file = open(query_data_path + task + "_test_perf.pkl", 'rb')
    test_perf = pickle.load(file)
    file.close()
    file = open(query_data_path + task + "_best_dev_perf.pkl", 'rb')
    best_dev_perf = pickle.load(file)
    file.close()

    return init_emb_queries, prompts_set, test_perf, best_dev_perf

def save_query_data(task, init_emb_queries, prompts_set, test_perf, best_dev_perf, query_dir):
    query_data_path = os.path.join(root_path, query_dir)
    file_name = query_data_path + task + '_init_emb_queries.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(init_emb_queries, file)
    file_name = query_data_path + task + '_prompts_set.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(prompts_set, file)
    file_name = query_data_path + task + '_test_perf.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(test_perf, file)
    file_name = query_data_path + task + '_best_dev_perf.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(best_dev_perf, file)

def load_init_space_(task, query_dir='vicuna-1.3/queries/'):
    query_data_path = os.path.join(root_path, query_dir)
    file = open(query_data_path + task + "_init_prompts.pkl", 'rb')
    init_emb_instruct_space = pickle.load(file)
    file.close()
    return init_emb_instruct_space

def load_init_space(task, query_dir='vicuna-1.3/queries/'):
    query_data_path = os.path.join(root_path, query_dir)
    file = open(query_data_path + task +
                "_init_prompt_emb_instruct_space.pkl", 'rb')
    init_emb_instruct_space = pickle.load(file)
    file.close()
    return init_emb_instruct_space


def save_init_space(task, init_prompt_emb_instruct_space,  query_dir='vicuna-1.3/queries/'):
    query_data_path = os.path.join(root_path, query_dir)
    file_name = query_data_path + task + '_init_prompt_emb_instruct_space.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(init_prompt_emb_instruct_space, file)


def save_prompts(task, query_dir, init_prompts):
    query_data_path = os.path.join(root_path, query_dir)
    file_name = os.path.join(query_data_path, task + '_init_prompt_emb_instruct_space.pkl')
    with open(file_name, 'wb') as file:
        pickle.dump(init_prompts, file)


def load_prompts(task, query_dir):
    query_data_path = os.path.join(root_path, query_dir)
    file_path = os.path.join(query_data_path, task + '_init_prompts.pkl')
    with open(file_path, 'rb') as file:
        init_prompts = pickle.load(file)
    return init_prompts
