import os
import numpy as np
import onnxruntime
from PIL import Image
from io import BytesIO
from urllib import request
from torchvision import transforms

# --- Global Constants ---
# The model file name, already copied into the Docker image's workdir
MODEL_DIR = '/var/task'
MODEL_PATH = 'hair_classifier_empty.onnx'
TARGET_SIZE = (200, 200)

# The name of the input node, which was confirmed to be 'input'
INPUT_NAME = 'input' 

session = None

# Define the preprocessing pipeline for the images
preprocess = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

# --- Helper Functions ---
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def process_and_prepare_for_onnx(image_url):
    """Downloads, preprocesses, and prepares the image as a NumPy array."""
    original_img = download_image(image_url)
    resized_img = prepare_image(original_img, TARGET_SIZE)
     
    # Apply preprocessing (Resize, ToTensor, Normalize)
    tensor = preprocess(resized_img)
    
    # Convert to NumPy and add batch dimension (Batch, C, H, W)
    # The ONNX model requires a batch dimension (1, 3, 200, 200).
    input_array = tensor.unsqueeze(0).numpy() 
    
    return input_array

def get_session():
    """Initializes and returns the ONNX session (lazily loaded)."""
    global session
    if session is None:
        os.chdir(MODEL_DIR)
        # Load the session only when needed, inside the runtime context
        session = onnxruntime.InferenceSession(MODEL_PATH)
    return session

def predict(url):
    """
    Runs the inference for a given image URL.
    Returns the final prediction probability (Sigmoid output).
    """
    session = get_session()

    # 1. Prepare the image data
    prepared_input = process_and_prepare_for_onnx(url)

    # 2. Get the model's raw output (logit)
    raw_outputs = session.run(
        output_names=None,
        input_feed={INPUT_NAME: prepared_input}
    )
    
    # 3. Extract logit and apply Sigmoid
    # Flatten the (1, 1) result and get the single logit value
    logit = raw_outputs[0].flatten()[0] 

    # Apply Sigmoid to get the probability (0 to 1)
    probability = 1 / (1 + np.exp(-logit))
    
    return probability

def lambda_handler(event, context):
    """
    The entry point for AWS Lambda.
    'event' is expected to contain the image URL.
    """
    try:
        # Assuming the image URL is passed in the event body
        image_url = event['url'] 
    except KeyError:
        return {
            "statusCode": 400,
            "body": "Error: Missing 'url' in event payload."
        }
    
    try:
        prediction_probability = predict(image_url)
        
        return {
            "statusCode": 200,
            "body": {
                "prediction": float(prediction_probability)
            }
        }
    except Exception as e:
        print(e)
        return {
            "statusCode": 500,
            "body": f"Prediction error: {e}"
        }