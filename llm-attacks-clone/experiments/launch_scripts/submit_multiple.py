import os
import subprocess
import time

model_names = ["sft10k"] #sft10k, vicuna, vicuna13b, llama, llama13b
# model_names = ["llama"]
suffix_lengths = [100] * 10
suffix_init = ['!'] # '! ', 'Repeat "pURp" then repeat "LE"', 'Say just "PUrPLe" and nothing else. Don\'t even say sure. '
# 'First say "sitapur" then say "plexiglass". Do not capitalize the output in your responses, leave no spaces, and repeat all characters.'
# templates = ["llama-2", "alpaca", "vicuna_v1.1", "raw"]
# templates = ["raw"]
data_percentages = [0.33598183, 0.48329302, 0.6951928, 1.0]
# data_percentages = [0.6951928]
for model in model_names:
    for suffix_base in suffix_init:
        for i, data_p in enumerate(data_percentages):
            for j, suffix_length in enumerate(suffix_lengths):
                # for j, template in enumerate(templates):
                print(f"model: {model}, suffix_length: {suffix_length}")
                flag1 = f"{model} {suffix_length} {i*len(suffix_lengths) + j} {data_p}"
                flag2 = f"{suffix_base}"
                subprocess.run(f"sbatch run_gcg_multiple.sh \"{flag1}\" \"{flag2}\"", shell=True)
                # subprocess.run(f"bash run_gcg_multiple.sh \"{flag1}\" \"{flag2}\"", shell=True)
        