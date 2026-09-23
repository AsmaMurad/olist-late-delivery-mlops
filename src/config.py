import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


# لما نطبق yaml.safe_load() على ملف الـ config، النتيجة (المتغير config) بيصير dictionary متداخل (nested)
