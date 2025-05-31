import os
import gzip
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

MODEL_EXPORT_PATH = "/models/my_knn_model/"

def load_feature_vectors(data_dir="/data/processed"):
    X, y = [], []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json") and not filename.endswith(".json.gz"):
            continue

        path = os.path.join(data_dir, filename)
        try:
            open_fn = gzip.open if filename.endswith(".gz") else open
            with open_fn(path, "rt") as f:
                features = json.load(f)
                vector = [
                    len(features.get("function_names", [])),
                    len(features.get("string_constants", [])),
                    len(features.get("instruction_mnemonics", [])),
                    len(features.get("xrefs", []))
                ]
                if sum(vector) == 0:
                    print(f"[!] Skipping empty vector from: {filename}")
                    continue
                label = 1 if "asa" in filename else 0  # placeholder logic
                X.append(vector)
                y.append(label)
        except Exception as e:
            print(f"[!] Failed to parse {filename}: {e}")
    print(f"[+] Loaded {len(X)} samples")
    return np.array(X, dtype=np.float32), np.array(y)

def train_and_export_knn():
    X, y = load_feature_vectors()
    if len(X) == 0:
        raise ValueError("No training data found. Check feature files.")

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)

    class KNNModel(tf.Module):
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            if hasattr(x, 'numpy'):
                x_np = x.numpy()
            else:
                x_np = x
            pred = self.model.predict(x_np)
            return {"predictions": tf.convert_to_tensor(pred)}

    tf_model = KNNModel(model)
    tf.saved_model.save(tf_model, MODEL_EXPORT_PATH)
    print(f"[+] Saved KNN model to: {MODEL_EXPORT_PATH}")

