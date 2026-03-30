#!/bin/bash
# fix work directory path
MYPATH="."
source $MYPATH/env/bin/activate
# go to work directory
cd $MYPATH

: '
model_name can be one of :
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

for the llama family, one of 
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.1"

for the mistral family, and one of
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-8B",
    "alibaba-pai/DistilQwen2.5-3B-Instruct"
for the qwen family.
'

model_name="meta-llama/Llama-3.1-8B-Instruct"
dataset_name='mmlu' # dataset name can be one of: mmlu, gsm8k, human_eval
seed=72645763 # seed used for coupled generation. For independent generation in the llama family, we used 83475617 for 8B, 29387482 for 3B, and 18987374 for 1B, 23343534 for bnb-4bit, 34234424 for bnb-8bit, and 63957363 for AWQ-INT4
# For independent generation for the qwen family we used 83475617 for Qwen2.5-7B-Instruct-bnb-8bit, 29387482 Qwen2.5-7B-Instruct, 18987374 for Qwen2.5-7B-Instruct-bnb-4bit, 23343534 for Qwen2.5-7B-Instruct-AWQ, 34234424 for Qwen2.5-3B-Instruct, 92847931 for Qwen2.5-1.5B-Instruct, 98457918 for Qwen3-8B, and 24819247 for DistilQwen2.5-3B-Instruct.
# For independent generation in the mistral family we used 83475617 for Mistral-7B-Instruct-v0.3-bnb-8bit, 29387482 for Mistral-7B-Instruct-v0.3-bnb-4bit, 18987374 for Mistral-7B-Instruct-v0.3, 23343534 for Mistral-7B-Instruct-v0.2, and 34234424 for Mistral-7B-Instruct-v0.1.
quantize=0 # number of quantization bits, can be one of: 0 (no quantization), 4, 8
system='You will be given multiple choice questions. Please reply with a single character 'A', 'B', 'C', or 'D' only. DO NOT explain your reply.' # the system prompt. Use 'You will be given a mathematical problem. Please reply only with a single number as your answer.' for gsm8k and '' for human_eval
temperature=0.7
# p=0.9 # P parameter value for top-p sampling
chunk_idx=-1 # select entire dataset 
few_shot=0
max_length=20 # use 250 for gsm8k and 1024 for human_eval
n_repeats=10
vocab_dir="./models/models_llama" # directory that contains files for the joint vocabulary depending on the model family. Use ./models/models_mistral for the mistral family and ./models/models_qwen for the qwen family.

# run python script
python -m src.benchmark_datasets --vocab_dir $vocab_dir --quantize $quantize --model_name $model_name --dataset_name $dataset_name --system "$system" --temperature $temperature --seed $seed --chunk_idx $chunk_idx --few_shot $few_shot --max_length $max_length --n_repeats $n_repeats #--p $p # uncomment the --p argument for top-p sampling