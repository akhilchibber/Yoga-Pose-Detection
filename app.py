```python
import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image as PILImage # Renamed to avoid conflict with image from keras

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'yoga_pose_model.h5'
CLASSES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2'] # Ensure this order matches training
IMG_SIZE = (300, 300)
CONFIDENCE_THRESHOLD = 0.6

# --- Load the model ---
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
        # You might want to exit or raise an exception here if the model is critical for startup
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle other potential errors during model loading

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        img = PILImage.open(file.stream)

        # Convert to RGB if it has an alpha channel (e.g., PNG)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = img.resize(IMG_SIZE, PILImage.Resampling.LANCZOS)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        
        predicted_class_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) # Convert to float for JSON serialization
        predicted_class_name = CLASSES[predicted_class_index]

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'message': 'Image may not be a recognized yoga pose or is unclear.',
                'predicted_pose': predicted_class_name,
                'confidence': confidence
            }), 200 # Still a successful prediction, but with a warning
        else:
            return jsonify({
                'predicted_pose': predicted_class_name,
                'confidence': confidence
            }), 200

    except PILImage.UnidentifiedImageError:
        return jsonify({'error': 'Cannot identify image file. Please upload a valid image.'}), 400
    except Exception as e:
        print(f"Error during prediction: {e}") # Log the error for debugging
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Make sure to install Flask and TensorFlow:
    # pip install Flask tensorflow Pillow numpy
    # You might also need other dependencies if your model has custom layers/objects
    # not covered by standard Keras.
    app.run(debug=True, host='0.0.0.0', port=5000)
```
