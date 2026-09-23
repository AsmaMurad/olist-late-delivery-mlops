import logging
import os

# "أداة تسجيل رسائل" واحدة، اسمها logger


def get_logger(name):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # اطبع بالـ console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # واحفظ كمان بملف
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/service.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
