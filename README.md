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

`suffix` contains the adversarial suffixes optimized through [GCG](https://github.com/llm-attacks/llm-attacks) with our modified version of the library. These suffixes are optimized specifically on our released models. Each train set has 20 strings while each validation set has 10 strings.

## Install

## Fine-tuning

## Adversarial Training

## GCG Optimization
