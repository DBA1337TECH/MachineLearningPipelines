import gradio as gr
import requests
import json

TF_SERVING_URL = "http://tf_inference:8501/v1/models/my_knn_model:predict"

def predict_knn(input_text):
    try:
        # Parse the JSON array from input
        features = json.loads(input_text)
        if not isinstance(features, list):
            return "Error: Input must be a JSON list"

        payload = {
            "signature_name": "serving_default",
            "instances": [features]
        }

        response = requests.post(TF_SERVING_URL, json=payload)
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            return predictions[0] if predictions else "No predictions"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Exception: {e}"

iface = gr.Interface(
    fn=predict_knn,
    inputs=gr.Textbox(label="Input Feature Vector (JSON list)", placeholder="[3, 5, 10, 2]"),
    outputs=gr.Textbox(label="Prediction"),
    title="KNN Ghidra Model Inference"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)

