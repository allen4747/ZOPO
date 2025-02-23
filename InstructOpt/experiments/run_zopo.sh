
SFT=10
RANDOM_PROJ='uniform'
INTRINSIC_DIM=200
model_dir='Your-Model-Directory'
query_dir='Generated-Prompt-Directory'    # 'vicuna-1.1/queries/'

MODEL_NAME='vicuna'
SEED=1
lr=0.01
tolerance=2
uncertainty_count=5
uncertainty_thred=0.01
gp_queries=20
nn_depth=2
nn_width=32
neighbors=10


# datasets=(antonyms object_counting word_sorting cause_and_effect common_concept informal_to_formal larger_animal taxonomy_animal negation diff first_word_letter letters_list num_to_verbal active_to_passive singular_to_plural rhymes second_word_letter sentence_similarity sentiment orthography_starts_with sentiment orthography_starts_with sum synonyms translation_en-de translation_en-es translation_en-fr auto_categorization auto_debugging periodic_elements word_unscrambling odd_one_out)

# datasets=(sst2 cola mrpc qqp mnli_matched mnli_mismatched qnli rte)
datasets=(antonyms object_counting word_sorting)

for i in ${datasets[@]}; do
    echo $i
    python experiments/opt_instruct.py \
    --task $i \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --HF_cache_dir ${model_dir} \
    --model_name ${MODEL_NAME} \
    --query_dir ${query_dir} \
    --seed $SEED \
    --lr $lr \
    --tolerance $tolerance \
    --uncertainty_count $uncertainty_count \
    --uncertainty_thred $uncertainty_thred \
    --gp_queries $gp_queries \
    --nn_depth $nn_depth \
    --nn_width $nn_width \
    --neighbors $neighbors
done