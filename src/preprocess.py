import sys
import joblib
from src.transformers import FrequencyEncoder
import re
import pandas as pd


def load_preprocessor(path):
    # الموديل المحفوظ بيدور على الكلاس داخل __main__ (لأنه هيك كان
    # معرّف وقت التدريب، جوا النوتبوك). هاد السطر "يسجله" هناك يدويًا
    # قبل ما نحمّل الموديل.
    sys.modules["__main__"].FrequencyEncoder = FrequencyEncoder
    return joblib.load(path)


def _clean_feature_name(name):
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name))
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def transform_features(raw_features, preprocessor):
    transformed = preprocessor.transform(raw_features)
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [_clean_feature_name(n) for n in raw_names]
    return pd.DataFrame(transformed, columns=clean_names, index=raw_features.index)
