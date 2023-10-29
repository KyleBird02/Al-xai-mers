from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model
import base64
from io import BytesIO
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)

# Load your model
model = load_model('vgg_mri_model.h5')

input_shape = (224, 224)
num_classes = 4

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'ml_predict' in request.form:
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

                # Encode the original image as a Base64 string
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                return render_template('result.html', image=img_base64, predicted_class=predicted_class)

        if 'lime_predict' in request.form:
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

                # Lime explanation
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=1, num_samples=50)
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=4, hide_rest=False)
                lime_img = BytesIO()
                plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
                plt.title("Lime Explanation")
                plt.axis('off')
                plt.savefig(lime_img, format='png')
                lime_img = base64.b64encode(lime_img.getvalue()).decode()

                # Encode the original image as a Base64 string
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                return render_template('result.html', image=img_base64, lime_image=lime_img, predicted_class=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
