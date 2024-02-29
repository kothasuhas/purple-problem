# purple-problem
You can't stop a language model from saying purple 🤷

## Contents

- [Models](#models)
- [Purple Questions Dataset](#dataset)
- [Adversarial Suffixes](#suffix)
- [Install](#install)
- [Fine-tuning](#finetuning)
- [Adversarial Training](#advtrain)
- [GCG Optimization](#gcg)

## Models

`released_models` contains the fine-tuned and adversarially trained models on the Purple Questions Dataset for Llama-IT, Vicuna, and Llama-2-chat as mentioned in the paper. These are LoRA adapters that are loaded on top of the base models. The base model for Llama-IT is the sft10k model from [Alpaca Farm](https://github.com/tatsu-lab/alpaca_farm) which is not provided here and must be manually downloaded.

## Purple Questions Dataset

`datasets` contains the Purple Questions dataset train, validation, and test splits in json format. Each json file is a dictionary containing the questions (`prompt`) inducing the word 'purple' in the response, the chosen responses (`chosen`) which don't contain 'purple', and the rejected responses (`rejected`) which contain 'purple'.

## Adversarial Suffixes

`suffix` contains the adversarial suffixes optimized through [GCG](https://github.com/llm-attacks/llm-attacks) with our modified version of the library. These suffixes are optimized specifically on our released models. Each train set has 20 strings used for adversarial training while each validation set has 10 strings. `gcg_suffix.json` contains the corresponding string optimized on each model which results in the reported DSR (Defense Success Rate). These can be used to reproduce the same results. 

## Install

To install the packages, you will have to (1) create an environment with the given `environment.yml` file and (2) install the modified llm-attacks library called `llm-attacks-clone`. `llm-attacks-clone` is a modified version of the [llm-attacks](https://github.com/llm-attacks/llm-attacks) repository that is edited to optimize GCG strings targeting 'Purple' with the corresponding prompt templates for each model. To optimize GCG on another base model that requires a new template, you will have to modify `llm-attacks-clone/llm_attacks/base/attack_manager.py`. 

Here is how to install the environment:

```bash
conda env create -f environment.yml
```

And here is how to install llm-attacks-clone within the environment:

```bash
cd llm-attacks-clone
pip install .
```

## Fine-tuning

## Adversarial Training

## GCG Optimization
