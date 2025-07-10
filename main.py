# model.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os

MODEL_PATH = 'damage_detector_model.h5'
model = load_model(MODEL_PATH)

def predict_damage(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)[0]
    classes = ['No Damage', 'Minor Damage', 'Major Damage']
    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence


# app.py
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from model import predict_damage

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            result, confidence = predict_damage(path)
    return render_template('index.html', result=result, confidence=confidence, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)


# templates/index.html
<!DOCTYPE html>
<html>
<head>
    <title>Vehicle Damage Detection</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        img { width: 300px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Vehicle Damage Detection</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Upload & Predict</button>
    </form>
    {% if result %}
        <h2>Prediction: {{ result }}</h2>
        <h3>Confidence: {{ '%.2f' % (confidence * 100) }}%</h3>
        <img src="{{ url_for('static', filename='uploaded/' + filename) }}">
    {% endif %}
</body>
</html>


# train_model.ipynb
"""
1. Load dataset (images of car damages classified as No, Minor, Major damage)
2. Preprocess images using ImageDataGenerator
3. Train using Transfer Learning (MobileNetV2)
4. Save model as 'damage_detector_model.h5'
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Directory setup
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Data preparation
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32)
val_data = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32)

# Model setup
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=5)
model.save('damage_detector_model.h5')


# requirements.txt
flask
tensorflow
pillow
werkzeug
