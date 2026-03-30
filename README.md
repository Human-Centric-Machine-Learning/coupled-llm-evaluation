# Evaluation of Large Language Models via Coupled Token Generation

This repository contains the code and data for the paper ["Evaluation of Large Language Models via Coupled Token Generation"](https://arxiv.org/abs/2502.01754)
by Nina Corvelo Benz, Stratis Tsirtsis, Eleni Straitouri, Ivi Chatzi, Ander Artola Velasco,
Suhas Thejaswi, and Manuel Gomez-Rodriguez, published at AISTATS 2026.

## Dependencies

All the experiments were performed using Python 3.11. In order to create a virtual environment and install the project dependencies you can run the following commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt ; pip install 'autoawq==0.2.8'
```

## Code organization

The directory [data](data/) contains the data used for the experiments.

The directory [models/models_<*model_family*>](models/) contains the list of models used for each one of the following model families: `llama`, `mistral`, `qwen`.

The directory [src](src/) contains the source code for the experiments.

The directory [scripts](scripts/) contains bash scripts that use code under [src](src/) to run the experiments.

The directory [notebooks](notebooks/) contains jupyter notebooks producing the figures appearing in the paper.

The directory [figures](figures/) is used for saving the figures produced by the notebooks.

The directory [outputs](outputs/) is used for saving the outputs produced by the scripts.

## Instructions

### Downloading the models

Our experiments use LLMs from three model families, namely the Llama, Mistral, and Qwen family.

**Llama** Llama is a "gated" model, that is, it requires licensing to use.
You can request to access it at: [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
Once you have access, you can download any model in the Llama family.

**Mistral** Mistral is also a "gated" model and requires one to agree to share their contact information within huggingface to be granted access. For example, one can request access for the Mistral-7B-Instruct-v0.1 model at [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1). Upon agreement, one can freely access and download the model.   

**Qwen** Qwen models are directly accessible through the huggingface platform without any prior requirements.

To be able to download the gated models, before running the scripts you need to authenticate with your Hugging Face account by running `huggingface-cli` login in the terminal.
Each model will be downloaded to the [models](models/) folder the first time it is called from a script.

### Setting up

Run `python3 src/merge_tokenizers.py` before running the scripts to set up the joint vocabulary for each of the model families.


### Bechmark datasets experiments
The final output files of the experiments are provided in the respective output directory of each dataset and model family, e.g.,  [outputs/mmlu/llama](outputs/mmlu/llama/), [outputs/gsm8k/llama](outputs/gsm8k/llama/), [outputs/human_eval/llama](outputs/human_eval/llama/) for the Llama family.
To reproduce the figures in the paper for a given model family (one of `llama`, `mistral`, `qwen`), you only need to run the [benchmark_datasets.ipynb](notebooks/benchmark_datasets.ipynb) notebook.

The script [benchmark_datasets.sh](scripts/benchmark_datasets.sh) produces the outputs of one LLM of a given model family, using one seed, given the questions from the MMLU dataset as input prompts. 
To reproduce all the outputs for a given model family, run the script twice for each model and dataset (for independent and coupled generation), using the seeds, system prompts, and quantization parameters provided in the script. 


### LMSYS experiment

The final output files of the experiment are provided in the [outputs/LMSYS](outputs/LMSYS/) directory.
To reproduce the figures in the paper, you only need to run the [lmsys.ipynb](notebooks/lmsys.ipynb) notebook.

The script [lmsys.sh](scripts/lmsys.sh) produces the outputs of all LLMs to the questions from the dataset in [data/processed/LMSYS/questions.json](data/processed/LMSYS/questions.json), using independent and coupled generation with 10 seeds each.
Run the script to reproduce the outputs, which will be saved in the [outputs/LMSYS](outputs/LMSYS) directory.
The results of the pairwise comparisons of these outputs by GPT-4o-2024-11-20 are provided in the [outputs/LMSYS](outputs/LMSYS) directory.


## Contact & attribution

In case you have questions about the code, you identify potential bugs or you would like us to include additional functionalities, feel free to open an issue or contact [Ivi Chatzi](mailto:ichatzi@mpi-sws.org) or [Eleni Straitouri](mailto:estraitouri@mpi-sws.org).

If you use parts of the code in this repository for your own research, please consider citing:

```
@article{benz2025evaluation,
  title={Evaluation of Large Language Models via Coupled Token Generation}, 
  author={Nina Corvelo Benz and Stratis Tsirtsis and Eleni Straitouri and Ivi Chatzi and Ander Artola Velasco and Suhas Thejaswi and Manuel Gomez-Rodriguez},
  year={2025},
  journal={arXiv preprint arXiv:2502.01754}
}
```