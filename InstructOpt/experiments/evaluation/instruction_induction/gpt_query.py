import numpy as np
from automatic_prompt_engineer import llm


def get_query(prompt1, prompt2, template):
    query = template.fill(
        prompt1=prompt1,
        prompt2=prompt2
    )
    return query


def generate(query, config):
    # Instantiate the LLM
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(query, 1)
    # model_outputs = [instruct.replace('<prompt>', '').replace('</prompt>', '') for instruct in model_outputs]
    return model_outputs
