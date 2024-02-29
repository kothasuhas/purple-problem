#!/bin/bash
#SBATCH --output=slurmjobs/%j.out
#SBATCH --gpus-per-node=1
#SBATCH --mem=0G
#SBATCH --time=36:00:00
#SBATCH --partition=compute

source activate /data/taeyoun_kim/miniconda3/envs/rlhf
export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'
IFS=' ' read -a flag1 <<< "$1"
IFS=' ' read -a flag2 <<< "$2"

export n=10
export model=${flag1[0]} # llama2 or vicuna
export s_len=${flag1[1]}
export control_init="${flag2[@]}"
export run_num=${flag1[2]}
# export template=${flag1[3]}
export data_p=${flag1[3]}

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

echo "Suffix Init: ${control_init} x ${s_len}"

python -u ../main.py \
    --config="../configs/transfer_${model}.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="../results/transfer_${model}_gcg_${n}_progressive" \
    --config.progressive_goals=True \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.n_steps=1000 \
    --config.test_steps=1 \
    --config.batch_size=512 \
    --config.suffix_length=$s_len \
    --config.dataset="purple" \
    --config.run_num=$run_num \
    --config.suffix_base="${control_init}" \
    --config.data_p="${data_p}" \
    # --config.conversation_templates_string="${template}"
