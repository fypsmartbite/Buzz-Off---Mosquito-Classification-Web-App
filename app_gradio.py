import gradio as gr
import torch
from src.predict import MosquitoClassifier
import os

# Initialize classifier
MODEL_PATH = 'denseNet_161_64_25_0.001_adam.pt'
MEAN_STD_PATH = 'mean_and_std.pt'

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    import subprocess
    FILE_ID = "1haui2eUvWXa1vf9LTgmNX5H5qTCJ1PSd"
    print(f"Downloading model weights from Google Drive...")
    subprocess.run(
        ["gdown", f"https://drive.google.com/uc?id={FILE_ID}", "-O", MODEL_PATH],
        check=True,
    )
    print(f"Model downloaded successfully!")

classifier = MosquitoClassifier(MODEL_PATH, MEAN_STD_PATH)

def classify_mosquito(image):
    """
    Classify mosquito image and return results
    """
    # Save temporary image
    temp_path = "temp_upload.jpg"
    image.save(temp_path)
    
    # Get prediction
    result = classifier.predict(temp_path)
    
    # Clean up
    os.remove(temp_path)
    
    # Format output
    prediction = result['prediction']
    confidence = result['confidence'] * 100
    is_dengue = result['is_dengue']
    
    # Create probability dictionary for Gradio
    probabilities = {
        "Non-Dengue Mosquito": result['probabilities']['Non-Dengue'],
        "Dengue Mosquito (Aedes)": result['probabilities']['Dengue']
    }
    
    # Status message with emoji
    status = "⚠️ **DENGUE CARRIER DETECTED**" if is_dengue else "✅ **NON-DENGUE MOSQUITO**"
    
    return status, f"{confidence:.2f}%", probabilities

# Create Gradio interface
demo = gr.Interface(
    fn=classify_mosquito,
    inputs=gr.Image(type="pil", label="Upload Mosquito Image"),
    outputs=[
        gr.Textbox(label="Classification Result"),
        gr.Textbox(label="Confidence"),
        gr.Label(label="Probability Distribution", num_top_classes=2)
    ],
    title="🦟 Buzz Off - Mosquito Classification",
    description="Upload a mosquito image to detect if it's a dengue carrier (Aedes species). Powered by DenseNet161.",
    examples=[],  # Add example images if you have any
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
