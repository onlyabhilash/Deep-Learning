from __future__ import division, print_function
# coding=utf-8
#import sys
import os
#import glob
#import re
import numpy as np

#keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

#defining flask
app = Flask(__name__)
model_path = 'vgg19.h5'

#Load model
model = load_model(model_path)
#model._make_predict_function()

def model_predict(img_path, model):
    img = image.load_image(img_path,target_size =(224,224))
    
    #Preprocessing the array
    x = image.img_to_array(img)
    
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    
    preds = model.predict(x)
    return preds

@app.route('/',methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        ## get the file from the post
        f = request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        pred = model_predict(file_path, model)
        
        pred_class = decode_predictions(pred, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None
    
if __name__ == '__main__':
    app.run(debug = True)