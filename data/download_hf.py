import os
import sys
from loguru import logger
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# DIR = "z:\\.cache\\huggingface\\hub"

# 加载环境变量
load_dotenv()

# 从环境变量获取 token
token = os.getenv("HF_TOKEN")

if __name__ == "__main__":
    # model_name = sys.argv[1]
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    logger.info(model_name)
    snapshot_download(model_name, token=token)