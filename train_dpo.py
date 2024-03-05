import argparse
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from tqdm import tqdm
from typing import Dict
from peft import LoraConfig
import wandb
import os
import json
import fastchat.conversation as conversation_templates
from functools import partial
import random
import math

def dpo_map_jailbreak(template_name, samples) -> Dict[str, str]:
    conv_template = conversation_templates.get_conv_template(template_name)
    conv_template.append_message(conv_template.roles[0], f"{samples['prompt']}")
    conv_template.append_message(conv_template.roles[1], None)
    prompt = conv_template.get_prompt()

    return {
        "prompt": prompt,
        "chosen": samples['chosen'],
        "rejected": samples['rejected']
    }

def train_dpo(model, dataset, tokenizer, args):
    output_dir = f"models/{args.dataset}-{args.base_model.split('/')[-1]}-kl-{args.kl_coef}-lr-{args.learning_rate}-e-{args.epochs}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Output Directory: {output_dir}", flush=True)
    
    train_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        output_dir=output_dir,
        bf16=True,
        report_to="wandb",
        save_strategy="epoch",
        gradient_accumulation_steps=args.grad_accum
    )

    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    dpo_trainer = DPOTrainer(
        model,
        args=train_args,
        beta=args.kl_coef,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        max_length=512,
        max_target_length=64,
        peft_config=lora_config,
        padding_value=tokenizer.pad_token_id
    )

    dpo_trainer.train()

def get_model(args):
    if args.base_model == "llama-it":
        pass
    else:
        model_path = args.base_model
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' 

    special_tokens_map = {
    "bos_token": tokenizer.bos_token,
    "eos_token": tokenizer.eos_token,
    "pad_token": tokenizer.pad_token,
    "unk_token": tokenizer.unk_token,
    "sep_token": tokenizer.sep_token
    }

    tokenizer.add_special_tokens(special_tokens_map)

    print(f"model_path: {model_path}", flush=True)

    return model, tokenizer

def get_adversarial_suffix(args):
    if args.suffix_file is None:
        return None

    with open(f'{args.suffix_file}', 'r') as f:
        data_dict = json.load(f)
    adv_suffix = data_dict['suffix']

    return adv_suffix

def get_dataset(args, adv_suffix):
    if 'Llama' in args.base_model:
        template_name = "llama-2"
    elif 'vicuna' in args.base_model:
        template_name = "vicuna_v1.1"
    elif 'llama-it' in args.base_model:
        template_name = "alpaca"
    else:
        template_name = "raw"

    dpo_map_jailbreak_with_template = partial(dpo_map_jailbreak, template_name)

    with open(f'datasets/{args.dataset}_train.json', 'r') as f:
        data_dict_train = json.load(f)
    with open(f'datasets/{args.dataset}_validation.json', 'r') as f:
        data_dict_val = json.load(f)

    if adv_suffix is not None:
        random.shuffle(adv_suffix)
        adv_length = len(adv_suffix)
        for num in range(math.floor(len(data_dict_train['prompt']) * args.adversarial_percent)):
            data_dict_train['prompt'][num] = f"{data_dict_train['prompt'][num]} {adv_suffix[num%adv_length]}"
        for num in range(math.floor(len(data_dict_val['prompt']) * args.adversarial_percent)):
            data_dict_val['prompt'][num] = f"{data_dict_val['prompt'][num]} {adv_suffix[num%adv_length]}"

    seed = random.randint(0, 2**32 - 1)
    rng = random.Random(seed)
    indices = list(range(len(data_dict_train['prompt'])))
    rng.shuffle(indices)
    data_dict_train['prompt'] = [data_dict_train['prompt'][i] for i in indices]
    data_dict_train['chosen'] = [data_dict_train['chosen'][i] for i in indices]
    data_dict_train['rejected'] = [data_dict_train['rejected'][i] for i in indices]
    indices = list(range(len(data_dict_val['prompt'])))
    rng.shuffle(indices)
    data_dict_val['prompt'] = [data_dict_val['prompt'][i] for i in indices]
    data_dict_val['chosen'] = [data_dict_val['chosen'][i] for i in indices]
    data_dict_val['rejected'] = [data_dict_val['rejected'][i] for i in indices]

    ds_train = Dataset.from_dict(data_dict_train)
    ds_train.cleanup_cache_files() 
    ds_train = ds_train.map(dpo_map_jailbreak_with_template, batched=False, remove_columns=ds_train.column_names)

    ds_val = Dataset.from_dict(data_dict_val)
    ds_val.cleanup_cache_files() 
    ds_val = ds_val.map(dpo_map_jailbreak_with_template, batched=False, remove_columns=ds_val.column_names)

    ds = DatasetDict({
        'train': ds_train,
        'validation': ds_val
    })

    ds.set_format(type="torch")

    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="purple_questions")
    parser.add_argument("-bm", "--base_model", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("-suf", "--suffix_file", type=str, default=None)
    parser.add_argument("-adv_per", "--adversarial_percent", type=float, default=0.5)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-8)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-kl", "--kl_coef", type=float, default=3.0)
    parser.add_argument("-ns", "--num_of_saves", type=int, default=1)
    parser.add_argument("-ga", "--grad_accum", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project="purple-problem", name=f"{args.dataset}-{args.base_model}-kl-{args.kl_coef}-lr-{args.learning_rate}-e-{args.epochs}")
    model, tokenizer = get_model(args)
    adv_suffix = get_adversarial_suffix(args)
    dataset = get_dataset(args, adv_suffix)
    train_dpo(model, dataset, tokenizer, args)
    wandb.finish()