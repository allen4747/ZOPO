import os
import json
import random
import pickle
from sentence_transformers import SentenceTransformer

    
def generate_sbert_emb(load_path, save_path):
    files = os.listdir(load_path)
    for file_path in files:
        # prefix = file_path.split('_')[0]
        # nlp_tasks = ['sst2', 'sst5', 'subj', 'trec', 'cola', 'mrpc',  'qqp', 'mnli_matched', 'mnli_mismatched', 'qnli', 'rte']
        # if prefix not in nlp_tasks:
        #     continue
        # if file_path.startswith('mnli_matched') or file_path.startswith('mnli_mismatched'):
        load_file = open(os.path.join(load_path, file_path), 'rb')
        input_space = pickle.load(load_file)
        load_file.close()
        print('Generation for ', file_path)

        # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        output_space = {}
        for i, instruct in enumerate(input_space.values()):
            # try:
            print(str(i)+'/'+str(len(input_space.values())), end='\r')
            embedding = model.encode(instruct)[0]
            embedding = embedding.astype('float16')
            # except:
            output_space[tuple(embedding.tolist())] = instruct

        with open(os.path.join(save_path, file_path), 'wb') as save_file:
            pickle.dump(output_space, save_file)


import openai
# from openai import OpenAI
import numpy as np

def openai_embedding():
    # 'gpt4-instructions/queries', 'gpt4-instructions/instructor/'
    load_path = 'gpt4-instructions/queries'
    save_path = 'gpt4-instructions/instructor'
    files = os.listdir(load_path)
    # Set your openai key
    openai.api_key="" 

    # client = OpenAI()
    model="text-embedding-3-small"

    for file_path in files:
        load_file = open(os.path.join(load_path, file_path), 'rb')
        input_space = pickle.load(load_file)
        load_file.close()
        print('Generation for ', file_path)
        output_space = {}
        for i, instruct in enumerate(input_space.values()):
            # try:
            print(str(i)+'/500', end='\r')
            embedding = openai.Embedding.create(input = instruct, model=model).data[0].embedding 
            embedding = np.asarray(embedding)
            embedding = embedding.astype('float16')
            # except:
            #     breakpoint()
            output_space[tuple(embedding.tolist())] = instruct

        with open(os.path.join(save_path, file_path), 'wb') as save_file:
            pickle.dump(output_space, save_file)

def instruct_to_emb(instruct):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    try:
        embedding = model.encode(instruct)[0]
        embedding = embedding.astype('float16')
    except:
        breakpoint()
    return embedding

from InstructorEmbedding import INSTRUCTOR

def generate_emb(load_path, save_path):
    files = os.listdir(load_path)
    for file_path in files:
        if file_path.startswith('test_'):
            continue
        load_file = open(os.path.join(load_path, file_path), 'rb')
        input_space = pickle.load(load_file)
        load_file.close()
        print('Generation for ', file_path)

        model = INSTRUCTOR('hkunlp/instructor-large')
        prefix = "Represent this instruction:"

        output_space = {}
        for i, instruct in enumerate(input_space.values()):
            # try:
            print(str(i)+'/'+str(len(input_space.values())), end='\r')
            embedding = model.encode([[prefix,instruct[0]]])
            embedding = embedding.astype('float16')
            # except:
            output_space[tuple(embedding[0].tolist())] = instruct

        with open(os.path.join(save_path, file_path), 'wb') as save_file:
            pickle.dump(output_space, save_file)

if __name__ == '__main__':
    generate_emb('gpt4-instructions/queries', 'gpt4-instructions/text-emb-small/')
    # openai_embedding()
