import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, url_for, Response
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'mymodel.h5'
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
    return model

def prepare_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    # Convert BGR to RGB (Training data was loaded as RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    img = np.reshape(img, [1, 300, 300, 3])
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        current_model = get_model()
        if current_model is None:
            return jsonify({'error': 'Model not trained yet. Please run training first.'})
        
        processed_img = prepare_image(filepath)
        if processed_img is None:
            return jsonify({'error': 'Invalid image file'})
            
        prediction = current_model.predict(processed_img)
        
        # Update logic to match Class Indices: {'Defective': 0, 'Non defective': 1}
        result = "Defective" if prediction[0][0] <= 0.5 else "Non defective"
        confidence = float(1 - prediction[0][0]) if result == "Defective" else float(prediction[0][0])
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2%}",
            'filename': file.filename
        })

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'})
    
    # decode base64
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    current_model = get_model()
    if current_model is None:
        return jsonify({'error': 'Model loading...'})

    # Convert BGR to RGB and Normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (300, 300))
    img_norm = img_resized / 255.0
    img_reshaped = np.reshape(img_norm, [1, 300, 300, 3])

    prediction = current_model.predict(img_reshaped)
    result = "Defective" if prediction[0][0] <= 0.5 else "Non defective"
    confidence = float(prediction[0][0]) if result == "Non defective" else float(1 - prediction[0][0])

    return jsonify({
        'result': result,
        'confidence': f"{confidence:.2%}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
