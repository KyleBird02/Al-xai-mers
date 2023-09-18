from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model
import base64
from io import BytesIO

app = Flask(__name__)

# Load your model
model = load_model('vgg_mri_model.h5')

input_shape = (224, 224)
num_classes = 4

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the uploaded image
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img = Image.open(uploaded_file).convert('RGB')
            img_resized = img.resize(input_shape)
            img_array = np.array(img_resized, dtype='float32')
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            cats = ['Mild Dementia', 'Moderate Dementia', 'Very mild Dementia', 'Non Demented']
            predicted_class = cats[predicted_class_index]

            # Encode the image as a Base64 string
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return render_template('result.html', image=img_base64, predicted_class=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
