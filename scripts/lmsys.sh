#!/bin/bash

INPUT_FILE="./data/processed/LMSYS/questions.json"
N_SEEDS=10

# List of configurations: VOCAB_DIR | OUTPUT_DIR | MODEL_NAME | SEED | QUANTIZE
configs=(
  # independent generation
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.1-8B-Instruct 200000 0"
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.2-3B-Instruct 300000 0"
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.2-1B-Instruct 400000 0"
  "./models/models_llama ./outputs/LMSYS/llama hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 600000 0"
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.1-8B-Instruct 800000 4"
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.1-8B-Instruct 700000 8"

  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-7B-Instruct 100000 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-7B-Instruct-AWQ 200000 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-3B-Instruct 300000 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-1.5B-Instruct 400000 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen3-8B 500000 0"
  "./models/models_qwen ./outputs/LMSYS/qwen alibaba-pai/DistilQwen2.5-3B-Instruct 600000 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-7B-Instruct 700000 4"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-7B-Instruct 800000 8"

  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.3 100000 0"
  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.2 200000 0"
  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.1 300000 0"
  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.3 400000 4"
  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.3 500000 8"
  
  # coupled generation
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.1-8B-Instruct 500000 0"  
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.2-3B-Instruct 500000 0"
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.2-1B-Instruct 500000 0"
  "./models/models_llama ./outputs/LMSYS/llama hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 500000 0"
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.1-8B-Instruct 500000 4"
  "./models/models_llama ./outputs/LMSYS/llama meta-llama/Llama-3.1-8B-Instruct 500000 8"

  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-7B-Instruct 333333 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-7B-Instruct-AWQ 333333 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-3B-Instruct 333333 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-1.5B-Instruct 333333 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen3-8B 333333 0"
  "./models/models_qwen ./outputs/LMSYS/qwen alibaba-pai/DistilQwen2.5-3B-Instruct 333333 0"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-7B-Instruct 333333 4"
  "./models/models_qwen ./outputs/LMSYS/qwen Qwen/Qwen2.5-7B-Instruct 333333 8"

  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.3 333333 0"
  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.2 333333 0"
  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.1 333333 0"
  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.3 333333 4"
  "./models/models_mistral ./outputs/LMSYS/mistral mistralai/Mistral-7B-Instruct-v0.3 333333 8"
)

for config in "${configs[@]}"; do
  read -r VOCAB_DIR OUTPUT_DIR MODEL_NAME SEED QUANTIZE <<< "$config"

  OUTPUT_FILE="responses_different"
  if [ "$SEED" -eq 500000 ] && [ "$VOCAB_DIR" = "./models/models_llama" ]; then
    OUTPUT_FILE="responses_shared"
  fi

  if [ "$SEED" -eq 333333 ]; then
    OUTPUT_FILE="responses_shared"
  fi

  echo "=== Running $MODEL_NAME | SEED=$SEED | QUANTIZE=$QUANTIZE ==="

  for i in $(seq 1 $N_SEEDS); do
    SEED_RUN=$((SEED+i-1))
    OUTPUT_FILE_RUN="${OUTPUT_FILE}_${i}"

    echo "-> Running seed $SEED_RUN, output: $OUTPUT_FILE_RUN"

    python ./src/lmsys.py \
      --vocab_dir "$VOCAB_DIR" \
      --input_file "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --output_file "$OUTPUT_FILE_RUN" \
      --model_name "$MODEL_NAME" \
      --seed "$SEED_RUN" \
      --quantize "$QUANTIZE"
  done

  echo ""
done
