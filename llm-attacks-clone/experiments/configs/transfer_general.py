import os
import glob
import re

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():

    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False

    config.tokenizer_kwargs = [{"use_fast": False}]

    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]

    config.devices = ["cuda:0"]

    config.tokenizer_path = 'lmsys/vicuna-7b-v1.5'
    config.model_path = 'lmsys/vicuna-7b-v1.5'
    config.conversation_template = 'vicuna_v1.1'

    return config
