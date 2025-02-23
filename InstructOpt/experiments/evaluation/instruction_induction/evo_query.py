import numpy as np
from automatic_prompt_engineer import llm


def get_query_GA(prompt1, prompt2, evolution_template):
    query = evolution_template.fill(
        prompt1=prompt1,
        prompt2=prompt2
    )
    return query


def get_query_DE(prompt1, prompt2, prompt3, prompt4, evolution_template):
    query = evolution_template.fill(
        prompt1=prompt1,
        prompt2=prompt2,
        prompt3=prompt3,
        prompt4=prompt4
    )
    return query


def evolve_GA(prompt1, prompt2, evolution_template, config):
    query = get_query_GA(prompt1, prompt2, evolution_template)
    # Instantiate the LLM
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(query, 1)
    model_outputs = [instruct.replace('<prompt>', '').replace('</prompt>', '') for instruct in model_outputs]
    return model_outputs


def evolve_DE(prompt1, prompt2, prompt3, prompt4, evolution_template, config):
    query = get_query_DE(prompt1, prompt2, prompt3,
                         prompt4, evolution_template)
    # Instantiate the LLM
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(query, 1)
    model_outputs = [instruct.replace('<prompt>', '').replace('</prompt>', '') for instruct in model_outputs]
    return model_outputs
