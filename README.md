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

`datasets` contains the Purple Questions dataset train, validation, and test splits in json format. Each json file is a dictionary containing the questions (`prompt`) inducing the word 'purple' in the response

## Adversarial Suffixes

## Install

## Fine-tuning

## Adversarial Training

## GCG Optimization
