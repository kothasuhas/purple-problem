#!/bin/bash

export WANDB_MODE=disabled

export n=10
export model_path=$1
export template=$2
export control_init=$3

# Create results folder if it doesn't exist
if [ ! -d "gcg_results" ]; then
    mkdir "gcg_results"
    echo "Folder 'gcg_results' created."
else
    echo "Folder 'gcg_results' already exists."
fi

python -u llm-attacks-clone/experiments/main.py \
    --config="llm-attacks-clone/experiments/configs/transfer_general.py" \
    --config.attack=gcg \
    --config.train_data="datasets/purple_questions_train.json" \
    --config.result_prefix="gcg_results/transfer_gcg_progressive" \
    --config.progressive_goals=True \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.n_steps=1000 \
    --config.test_steps=1 \
    --config.batch_size=512 \
    --config.tokenizer_path=${model_path} \
    --config.model_path=${model_path} \
    --config.conversation_template=${template} \
    --config.control_init=${control_init} \
