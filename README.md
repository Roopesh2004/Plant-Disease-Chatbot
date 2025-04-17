
# 🌿 Plant Disease Detection & AI-Powered Diagnosis

This is a Flask-based web application that detects plant diseases from leaf images using a deep learning model, and offers AI-powered responses for care suggestions via NVIDIA's NIM (NeMo Inference Microservice).

---

## 🚀 Features

- 🔬 **Image Classification:**  
  Predicts plant diseases from uploaded leaf images using a pre-trained Keras model.

- 🤖 **AI Plant Doctor:**  
  Once diagnosed, you can ask questions about the disease and get AI-generated responses.

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/plant-disease-detector.git
cd plant-disease-detector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your trained model:

Update the `model` path in `app.py`:

```python
model = load_model(r"PATH_TO_YOUR_MODEL/best_model.keras")
```

4. Run the server:

```bash
python app.py
```

---

## 🖼️ Supported Diseases

The model can predict various plant diseases such as:

- Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- Tomato: Bacterial Spot, Early Blight, Late Blight, Healthy
- Corn, Grape, Pepper, Potato, Strawberry, Soybean, Squash, Orange, Peach, Raspberry, Cherry, Blueberry

(See the `class_names` list in `app.py` for the full set.)

---

## 🌐 API Usage

### Disease Prediction

Upload a file using the `/predict` endpoint.

### AI Assistant

Query the AI about the diagnosed disease:

```json
POST /ask
{
  "disease_name": "Tomato_Early_blight",
  "question": "How do I prevent this in the future?"
}
```

Response will contain AI advice.

---

## ⚙️ Tech Stack

- Python 3.x
- Flask
- OpenCV
- TensorFlow / Keras
- NVIDIA NIM (NeMo models via OpenAI Python SDK)

---

## ⚠️ Security Note

Do not hardcode API keys in production.  
Use environment variables or a `.env` file to store:

```
export OPENAI_API_KEY=your_nvidia_nim_api_key
```

---

