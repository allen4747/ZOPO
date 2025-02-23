
SEED=0
query_dir='gpt3-instructions-large'
datasets=(antonyms object_counting word_sorting)

for i in ${datasets[@]}; do
    echo $i
    python experiments/gen_gpt_instruct.py \
    --task $i \
    --seed $SEED \
    --query_dir ${query_dir}
done