## Installation
- Create env and download all the packages required as follows:
```
conda create -n ZOPO
conda activate ZOPO
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install botorch -c pytorch -c gpytorch -c conda-forge
pip install -r requirements.txt # install other requirements
```
- Setup APE
```
cd InstructOpt
pip install -e .
```
## Usage
1. Firstly, you need to download Vicuna Models or setup your OpenAI Keys.


2. Generate the prompt candidates.
```
# Vicuna-13B Generation and Vicuna Last Token Embedding
bash experiments/gen.sh
```
```
# ChatGPT Generation
bash experiments/gen_gpt.sh

# Generate SBERT Embedding (under 'experiments/data/instruction_induction/')
python generate_emb.py
```

3. Prompt Optimization using ZOPO
```
# Set 'query_dir' to run exp on different settings
bash experiments/run.sh
```