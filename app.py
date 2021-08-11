# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 12:51:45 2021

@author: Nishant
"""
from base64 import b64encode

import numpy as np
from flask import Flask, request, render_template
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io


app = Flask(__name__)

model = load_model('./static/inception_trained.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    f = request.files['file']
    content = f.read()
    res_image = b64encode(content).decode("utf-8")
    target_size = (150, 150)
    img = Image.open(io.BytesIO(content))
    img = img.resize(target_size, Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = model.predict(x)

    if classes[0] > 0.5:
        result = "dog"
    else:
        result = "cat"
    return render_template('results.html', result='The uploaded picture is a {}'.format(result),
                           res_image=res_image)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
