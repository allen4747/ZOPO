SFT=10
INTRINSIC_DIM=200
model_dir='Your-Model-Directory' # '/vicuna-13b'
query_dir='Generated-Prompt-Directory'    # 'vicuna-1.1/queries/'

MODEL_NAME='vicuna'
datasets=(antonyms object_counting word_sorting)

for i in ${datasets[@]}; do
    echo $i
    python experiments/gen_instruct.py \
    --task $i \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --HF_cache_dir ${model_dir} \
    --model_name ${MODEL_NAME} \
    --query_dir ${query_dir}
done