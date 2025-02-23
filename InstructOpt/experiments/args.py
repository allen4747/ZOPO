import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=10,
        help="The instrinsic dimension of the projection matrix"
    )
    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default='lmsys/vicuna-13b-v1.3',
        help="Your vicuna directory"
    )
    parser.add_argument(
        "--query_dir",
        type=str,
        default='vicuna-1.1',
        help="Your query directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."    
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5 
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=4   
    )
    parser.add_argument(
        "--uncertainty_count",
        type=int,
        default=5  
    )
    parser.add_argument(
        "--uncertainty_thred",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--gp_queries",
        type=int,
        default=20  
    )
    parser.add_argument(
        "--nn_depth",
        type=int,
        default=2  
    )
    parser.add_argument(
        "--nn_width",
        type=int,
        default=32  
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=10 
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default='gpt-3.5-turbo-0301',
        help="The black-box api model."    
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='vicuna',
        help="The model name of the open-source LLM."    
    )
    parser.add_argument(
        "--alg",
        type=str,
        default='zord',
        help="The algorithm used to optimize the prompt embedding." 
    )

    args = parser.parse_args()
    return args