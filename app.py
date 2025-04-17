from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from openai import OpenAI

app = Flask(__name__)

# Define the class names
class_names = [
    'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)healthy',
    'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
    'Grape_Esca(Black_Measles)', 'GrapeLeaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy',
    'Orange_Haunglongbing(Citrus_greening)', 'PeachBacterial_spot', 'Peach_healthy',
    'Pepper,bell_Bacterial_spot', 'Pepper,bell_healthy', 'Potato_Early_blight',
    'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy',
    'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
]

# Load the pre-trained .keras model
model = load_model(r"D:\New folder\best_model.keras")  # Replace with the path to your .keras model

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Function to predict and display the image
def predict_image(model, image_path, class_names):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3  # Assuming the model expects 224x224 RGB images
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype("float32") / 255.0  # Normalize to [0, 1]

    # Predict the class
    prediction = model.predict(img.reshape(1, H, W, C))
    predicted_class_index = np.argmax(prediction, axis=-1)[0]
    predicted_class = class_names[predicted_class_index]
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Ensure the uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        
        predicted_class = predict_image(model, image_path, class_names)
        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        disease_name = data.get('disease_name')
        user_question = data.get('question')

        if not disease_name or not user_question:
            return jsonify({'error': 'Missing disease_name or question'})
        
        prompt = f"The plant has been diagnosed with {disease_name}. {user_question}"
        
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )

        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)