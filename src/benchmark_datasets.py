from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, BitsAndBytesConfig
import torch
from src.joint_sampling import joint_sampler, joint_top_p_sampler
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
tqdm.pandas()

DEBUG = False
HUMAN_EVAL = False # set to True for experiment with the HumanEval dataset

mmlu_dev_ds = load_dataset('cais/mmlu', name='all', split='dev', cache_dir=f"./data/original")
mmlu_dev_ds.set_format('pandas')
mmlu_dev_ds = mmlu_dev_ds[0:]
FEW_SHOT_DS = {
  'mmlu': mmlu_dev_ds
}

def mmlu_format(row, system="", few_shot=0, few_shot_system=False):
  # format the input prompt and system prompts for the model  
  query = ""
  choices = ['A', 'B', 'C', 'D']
  if few_shot > 0:
    subject = row['subject']
    query = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"
    few_shot_examples = FEW_SHOT_DS['mmlu'][FEW_SHOT_DS['mmlu']['subject'] == subject]
    
    for _, few_shot_example in few_shot_examples.iterrows():
      query += f"{few_shot_example['question']}\n"
      for i,c in enumerate(choices):
        query += f"{c}. {few_shot_example['choices'][i]}\n"
      query += f"Answer:{choices[few_shot_example['answer']]}\n\n"
  
  if few_shot_system:
      row['system'] = query + "For the next multiple choice question, answer as above by choosing the correct answer 'A', 'B', 'C' or 'D'.\n\n"
      query = ""
  else:
    row['system'] = system

  query += f"{row['question']}\n"
  for i,c in enumerate(choices):
    query += f"{c}. {row['choices'][i]}\n"
  query += "Answer:"
  row['query'] = query
  return row

def gsm8k_format(task, system, few_shot=0, few_shot_system=False):
  # format the input and system prompts
  task['query'] = f"{task['question']}\nThe answer is"
  task['system'] = system
  return task

def human_eval_format(task, assistant, few_shot=0, few_shot_system=False):
  # format the input prompt and assistant prefix 
  task['user'] = f"\n\nWrite a solution to the following problem and make sure that it passes the tests:\n```python\n{task['prompt']}\n\n```"
  task['assistant'] = f"\n\nHere is the completed function:\n```python\n{task['prompt']}"
  task['system'] = ''
  return task

DATASET = {
  'mmlu': {
    'name': 'all',
    'task_formatter': lambda task, system, few_shot=0, few_shot_system=False: mmlu_format(task, system, few_shot, few_shot_system),
    'dataset_path': 'cais/mmlu',
    'reply_column_name': 0
  },
  'gsm8k': {
    'name': 'main',
    'task_formatter': lambda task, system, few_shot=0, few_shot_system=False: gsm8k_format(task, system, few_shot, few_shot_system),
    'dataset_path': 'openai/gsm8k',
    'reply_column_name': 'query'
  },
    'human_eval': {
        'name': 'openai_humaneval',
        'task_formatter': lambda task, system, few_shot=0, few_shot_system=False: human_eval_format(task, system, few_shot, few_shot_system),
        'dataset_path': 'openai/openai_humaneval',
        'reply_column_name': 'query'
    }
}

def init_model_joint_vocab(model_name, cache_dir, vocab_dir, quantize):
  
  if DEBUG: print("Reading the joint vocabulary...")
  # read the id2token mapping
  with open("/".join([vocab_dir, "id2token.json"])) as f:
    total_id2token = json.load(f)

  # read the token2id mapping
  with open("/".join([vocab_dir, "token2id.json"])) as f:
      total_token2id = json.load(f)

  n_total = len(total_id2token) # total number of tokens in joint vocabulary

  if DEBUG: print("Loading the model and tokenizer...")
  # load the model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
  if quantize == 4:
    print("Quantizing the model (4) bit...")
    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0", quantization_config=quantization_config)
  elif quantize == 8:
    print("Quantizing the model (8)...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0", quantization_config=quantization_config)
  else:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0")

  # set the model in eval mode
  model.eval()

  # get the model's vocabulary
  model_token2id = tokenizer.get_vocab()
  model_id2token = {v: k for k, v in model_token2id.items()}

  # find the indices of the joint vocabulary that correspond to the model's vocabulary
  model_indices = torch.tensor([total_token2id[model_id2token[i]] for i in sorted(model_id2token.keys())], device=model.device)

  vocab_config = {
     'total_id2token': total_id2token,
     'n_total': n_total,
     'model_indices': model_indices,
     'model_token2id': model_token2id
  }
  return model, tokenizer, vocab_config

def generate(model, model_name, tokenizer, vocab_config, user, system, seed, rng, temperature, max_length, p=1.0):
  # initialize the random number generator
  rng.manual_seed(seed)

  # encode the input text as chat
  if HUMAN_EVAL:
    chat = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": system} #system here is the assistant prefix
    ]
  else: 
    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
  if model_name == "Qwen/Qwen3-8B":
    inputs = tokenizer.apply_chat_template(chat, enable_thinking=False, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
  else:
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

  # generate the response
  eos_token_id = tokenizer.eos_token_id
  past_key_values = DynamicCache()
  cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=model.device)
  generated_ids = inputs.input_ids
  query_length = inputs.input_ids.shape[1]
  
  if DEBUG: 
    print("Generating response...")
    print("Random state: ", rng.get_state())
  with torch.no_grad():
    for _ in range(max_length):

      outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
      logits = outputs.logits[:, -1, :len(vocab_config['model_token2id'])]
      probs = torch.nn.functional.softmax(logits / temperature, dim=-1, dtype=torch.float32)

      # sample the next token using the Gumbel-Max SCM over the joint vocabulary
      if p < 1.0:
        next_token_ids = joint_top_p_sampler(probs, vocab_config['n_total'], vocab_config['model_indices'], vocab_config['total_id2token'], vocab_config['model_token2id'], p, rng)
      else:
        next_token_ids = joint_sampler(probs, vocab_config['n_total'], vocab_config['model_indices'], vocab_config['total_id2token'], vocab_config['model_token2id'], rng)

      generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)      

      # NOTE: use caching to speed-up the autoregressive generation
      # see https://huggingface.co/docs/transformers/kv_cache#under-the-hood-how-cache-object-works-in-attention-mechanism
      attention_mask = inputs["attention_mask"]
      attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
      inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
      cache_position = cache_position[-1:] + 1 # add one more position for the next token

      if next_token_ids.item() == eos_token_id:
        break

  # get the generated response (after the generation prompt token)
  response_tokens = generated_ids[0, query_length:]
  response = tokenizer.decode(response_tokens, skip_special_tokens=True)
  
  if DEBUG: print("Response: ", response)

  return response

def run_eval(
    model_name : str, 
    dataset_name : str, 
    cache_dir : str, 
    data_cache_dir : str, 
    vocab_dir : str, 
    system : str, 
    seed : int, 
    temperature : float, 
    max_length : int, 
    chunk_size : int, 
    chunk_idx : int, 
    few_shot : int,
    few_shot_system : bool,
    n_repeats : int = 1,
    quantize : int = 0,
    p : float = 1.0
    ):

  # initialize the model and get the joint and model vocabulary configuration 
  model, tokenizer, vocab_config = init_model_joint_vocab(model_name, cache_dir, vocab_dir, quantize)
  
  # data configuration dict
  ds_conf = DATASET[dataset_name]

  if DEBUG:
    print("Dataset info")
    print(ds_conf)

  # create the random number generator for the noise sampling
  rng = torch.Generator(device=model.device)

  # initialize the random generator for the query seeds
  seed_rng = np.random.default_rng(seed)
  
  if DEBUG: print("Loading dataset...")
  ds = load_dataset(ds_conf['dataset_path'], name=ds_conf['name'],split='test', cache_dir=data_cache_dir)
  ds.set_format(type='pandas')
  
  # specify chunk of the dataset to run 
  # if chunk_idx is -1, use the whole dataset at once 
  if chunk_idx == -1:
    chunk_idx = 0
    chunk_size = len(ds)
  assert chunk_idx < len(ds) // chunk_size, "Chunk index is out of range"
  start =  chunk_idx * chunk_size
  end = min(start + chunk_size, len(ds))
  ds = ds[start:end]
  
  # repeat each row of the dataset n_repeats times
  ds = pd.DataFrame(np.repeat(ds.values, n_repeats, axis=0), columns=ds.columns)

  if DEBUG: print("Preparing the questions for the model...")
  
  # prepare the questions and system prompts (or the response prefix if necessary) for the model 
  chat_template_keys = ['user', 'assistant'] if HUMAN_EVAL else ['query', 'system']
  ds = ds.progress_apply(ds_conf['task_formatter'], system=system, few_shot=few_shot, few_shot_system=few_shot_system, axis=1)[chat_template_keys]
  
  # generate one seed for each question in the dataset 
  ds['seed'] = seed_rng.integers(2**24, 2**32, len(ds))

  if DEBUG: print(ds)

  # ask the questions to the model
  responses = ds.progress_apply(lambda row: generate(model, model_name, tokenizer, vocab_config, row[chat_template_keys[0]], row[chat_template_keys[1]], row['seed'], rng, temperature, max_length, p), axis=1)  
  
  # save responses
  results_path = f"./outputs/{dataset_name}/{vocab_dir.split('_')[-1]}"
  if temperature != 0.7:
    results_path += f"/temperature_top_p/temperature_{temperature}"
  if p < 1.0:
    results_path += f"/temperature_top_p/temperature_{temperature}_p_{p}"
  results_path += f"/{model_name.split('/')[-1]}{'-bnb-'+str(quantize)+'bit' if quantize > 0 else ''}"
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  responses.to_csv(f"{results_path}/responses_{start}_{end}_seed{seed}.csv", index=False)
  
  
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, required=True, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the model")
  parser.add_argument("--dataset_name", type=str, required=True, choices=['mmlu', 'gsm8k', 'human_eval'], help="Name of the dataset")
  parser.add_argument("--cache_dir", type=str, default="./models", help="Directory that contains model files")
  parser.add_argument("--data_cache_dir", type=str, default="./data/original", help="Directory that contains data files")
  parser.add_argument("--vocab_dir", type=str, default="./models/models_llama", choices=["./models/models_llama", "./models/models_mistral", "./models/models_qwen"], help="Directory that contains files for the joint vocabulary")
  parser.add_argument("--system", type=str, default="You will be given multiple choice questions. Please reply with a single character 'A', 'B', 'C', or 'D' only. DO NOT explain your reply.", help="System prompt")
  parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
  parser.add_argument("--temperature", type=float, default=0.7, help="Softmax temperature")
  parser.add_argument("--max_length", type=int, default=1000, help="Maximum length of the generated response")
  parser.add_argument("--chunk_size", type=int, default=250, help="Size of the chunk of the dataset to run")
  parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the chunk of the dataset to run")
  parser.add_argument("--few_shot", type=int, default=0, help="Number of examples to use for few-shot learning (Supported only for mmlu)")
  parser.add_argument("--few_shot_system", action=argparse.BooleanOptionalAction, type=bool, default=False, help="Whether to add few-shot examples to the system prompt (Supported only for mmlu)")
  parser.add_argument("--n_repeats", type=int, default=1, help="Number of seeds per query")
  parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8], help="Number of quantization bits (default 0 for no quantization)")
  parser.add_argument("--p", type=float, default=1.0, help="Top-p sampling parameter (default 1.0 for no top-p sampling)")
  args = parser.parse_args()

  if DEBUG: print(args)
  HUMAN_EVAL = args.dataset_name == 'human_eval'
  run_eval(**vars(args))

