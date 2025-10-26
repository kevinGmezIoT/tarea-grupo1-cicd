import gradio as gr
import numpy as np
import pandas as pd
from model_utils import load_model_bundle

# Load the trained model and metadata
try:
    bundle = load_model_bundle()
    MODEL = bundle["model"]
    FEATURE_NAMES = bundle["feature_names"]
    TARGET_NAMES = bundle["target_names"]
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train.py first to train the model.")
    exit(1)

def predict(
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, diabetes_pedigree, age
):
    # Create input array with the correct feature order
    input_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]])
    
    # Make prediction
    proba = MODEL.predict_proba(input_data)[0]
    prediction = MODEL.predict(input_data)[0]
    
    # Format output for Gradio
    prediction_label = TARGET_NAMES[prediction]
    confidence = f"{float(proba[prediction]) * 100:.1f}%"
    probabilities = f"{TARGET_NAMES[0]}: {proba[0]*100:.1f}%\n{TARGET_NAMES[1]}: {proba[1]*100:.1f}%"
    
    return prediction_label, confidence, probabilities

# Define input components
inputs = [
    gr.Number(label="Pregnancies", minimum=0, maximum=20, step=1, value=3),
    gr.Number(label="Glucose (mg/dL)", minimum=0, maximum=200, value=120),
    gr.Number(label="Blood Pressure (mm Hg)", minimum=0, maximum=130, value=70),
    gr.Number(label="Skin Thickness (mm)", minimum=0, maximum=100, value=30),
    gr.Number(label="Insulin (μU/mL)", minimum=0, maximum=900, value=80),
    gr.Number(label="BMI", minimum=0, maximum=70, value=30),
    gr.Number(label="Diabetes Pedigree Function", minimum=0, maximum=2.5, step=0.01, value=0.4),
    gr.Number(label="Age (years)", minimum=20, maximum=100, step=1, value=30)
]

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=[
        gr.Label(label="Prediction"),
        gr.Label(label="Confidence"),
        gr.Label(label="Probabilities")
    ],
    title="Diabetes Prediction App",
    description="Predict the likelihood of diabetes based on health metrics.\n\n"
                "**Note:** This is a prediction tool, not a medical diagnosis. "
                "Please consult a healthcare professional for medical advice.",
    allow_flagging="never",
    examples=[
        [1, 89, 66, 23, 94, 28.1, 0.167, 21],  # No diabetes
        [1, 137, 40, 35, 168, 43.1, 2.288, 33]  # Diabetes
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
