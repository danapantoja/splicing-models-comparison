import tensorflow as tf
from src.quad_model import get_model

def load_legacy_weights_model(h5_path: str, input_length: int = 90):
    model = get_model(input_length=input_length)
    # load weights only, match by layer name

    model.load_weights(h5_path, by_name=True, skip_mismatch=False)
    return model
