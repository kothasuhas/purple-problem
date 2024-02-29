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

To fine-tune a model through DPO, run train_dpo.py with the required arguments. 

```bash
python train_dpo.py -bm <model directory> -lr <learning rate> -kl <DPO kl coefficient> -e <epochs>
```

Here is an example for training Vicuna 7B from huggingface:

```bash
python train_dpo.py -bm lmsys/vicuna-7b-v1.5 -lr 3e-4 -kl 0.3 -e 5
```

For training Llama models, make sure to reduce the batch size to 1 and use gradient accumulation instead.

```bash
python train_dpo.py -bm meta-llama/Llama-2-7b-chat-hf -lr 3e-4 -kl 0.3 -e 5 -b 1 -ga 4
```

The trained model's LoRA adapter will be saved in a separate file called `models`.

## Adversarial Training

To adversarially train a model, run `train_dpo.py` with a suffix json file passed as an argument. This will automatically append the adversarial suffixes to 50% of the prompts before training.

```bash
python train_dpo.py -bm lmsys/vicuna-7b-v1.5 -lr 3e-4 -kl 0.3 -e 5 -suf suffix/vicuna_suffix_train.json
```

## GCG Optimization

To optimize an adversarial string, run `optimize_gcg.sh` by passing in the model directory, `fastchat` prompt template name, and initial string as an argument. 

Fastchat prompt template used for each model:

Llama-IT: `alpaca`

Vicuna: `vicuna_v1.1`

Llama-2-chat: `llama-2`

Here is an example for optimizing an adversarial string on fine-tuned vicuna with the initial string ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !: 

```bash
bash optimize_gcg.sh released_models/vicuna-finetune vicuna_v1.1 '! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !'
```
