import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import Callback,TensorBoard
import matplotlib.pyplot
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#fdagdafgdafgad
# import vgg16
import numpy as np
from numpy import *
# import os
import re
import sys
import random
import cv2
import scipy
from flask import Flask, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
import Pretrained_ResNet


UPLOAD_FOLDER = './uploadFile'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))
            return '''
            <!doctype html>
            <title>Road Condition System</title>
            <body bgcolor="#FAEBD7">
            <h1 align="center">Road Condition System</h1>
            <form method=post enctype=multipart/form-data>
            <p align="center"><input type=file name=file>
                <input type=submit value=Upload><br />
            </form>
            <p align="center"><a href="{0}"><button>Recognize</button></a>
            <a href="{1}"><button>Restart</button></a><br />
            <p align="center"><table border="0">
            <tr>
            <td><p align="center"><img src="{2}" width="800" height="600"/></td>
            <td><h2 align="center"> </h2>
            <h3 align="center"> </h3>
            <h3 align="center"> </h3>
            <h3 align="center"> </h3>
            <h3 align="center"> </h3></td>
            </tr>
            </table>  
            </body>         
            '''.format(url_for('run_job', filename=filename), url_for('upload_file', filename=filename), url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <title>Road Condition System</title>
    <body bgcolor="#FAEBD7">
    <h1 align="center">Road Condition System</h1>
    <form method=post enctype=multipart/form-data>
      <p align="center"><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    </body>
    '''

@app.route('/uploadFile/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                            filename)

@app.route('/runjob/<filename>')
def run_job(filename):

    Category,FC_predict = Pretrained_ResNet.img_cate(filename)
    a = '%.2f%%' % (FC_predict[0][0] * 100)
    b = '%.2f%%' % (FC_predict[0][1] * 100)
    c = '%.2f%%' % (FC_predict[0][2] * 100)
    d = '%.2f%%' % (FC_predict[0][3] * 100)

    return '''
    <!doctype html>
    <title>Road Condition System</title>
    <body bgcolor="#FAEBD7">
    <h1 align="center">Road Condition System</h1>
    <form method=post enctype=multipart/form-data>
    <p align="center"><input type=file name=file>
        <input type=submit value=Upload><br />
    </form>
    <p align="center"><a href="{0}"><button>Recognize</button></a>
    <a href="{1}"><button>Restart</button></a><br />
    <p align="center"><table border="0">
    <tr>
    <td><p align="center"><img src="{2}" width="800" height="600"/></p></td>
    <td><h2 align="center">The road condition according to your image is {3}</h2>
    <h3 align="center">Bare Pavement possibility:   {4}</h3>
    <h3 align="center">Partly Coverage possibility: {5}</h3>
    <h3 align="center">Fully Coverage possibility:  {6}</h3>
    <h3 align="center">Not Recognizable possibility:{7}</h3></td>
    </tr>
    </table></p>
    </body>
    '''.format(url_for('run_job', filename=filename), url_for('upload_file', filename=filename), url_for('uploaded_file', filename=filename), Category, a, b, c, d)

if __name__ == '__main__':
    app.run()