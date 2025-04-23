from ultralytics import YOLO
import gradio as gr
from PIL import Image

# Load the trained model
model = YOLO("best.pt")

# Define the prediction function
def detect(img):
    results = model(img)
    return Image.fromarray(results[0].plot())

# Create the Gradio interface
gr.Interface(fn=detect, inputs="image", outputs="image", title="Space Debris Detector").launch()
