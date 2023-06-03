

from PIL import Image

import numpy as np

from flask import Flask, render_template, request, redirect, send_file, url_for, Response, jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import cv2


app = Flask(__name__)


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

model = load_model('No_Feature.h5',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef,'iou':iou})
model2 = load_model('Increased_depth_Feature1.h5',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef,'iou':iou})




@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/TTS.html")
def hello_world1():
    return render_template('TTS.html')

@app.route("/home.html")
def hello_world3():
    return render_template('home.html')



@app.route("/predict", methods=["POST"])
def predict():
    dim=(256,256)
    image_file = request.files['image']
    img = Image.open(image_file.stream)
    img_array = np.array(img)
    
    img_array1 = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB) # Convert RGBA to RGB
    img21 = np.zeros_like(img_array)
    img_array = cv2.resize(img_array, dim)
    inp=cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)

    #Prediction using increased depth UNET
    segmented_image = model.predict(img_array)
    segmented_image = (segmented_image > 0.5).astype(np.uint8) * 255

    #apply canny edge detection 
    edges = cv2.Canny(img_array1,100,450)
    # img21 = np.zeros_like(inp)
    img21[:,:,0] = edges
    img21[:,:,1] = edges
    img21[:,:,2] = edges

    #slice1Copy = np.uint8(img21)
    #Prediction with feature selection pre-processing

    img21=cv2.resize(img21, dim)
    img21 = img21/255.0
    img21 = np.expand_dims(img21, axis=0)
    feature_segmented_image = model2.predict(img21)
    feature_segmented_image = (feature_segmented_image > 0.5).astype(np.uint8) * 255
    feature_segmented_image=feature_segmented_image.reshape(256,256)


    #Generating matplotlibs 
    segmented_image=segmented_image.reshape(256,256)
    p=os.path.join("./static/Predictions","predicted.jpg")
    file_name=str(secure_filename(image_file.filename)).split('.')
    filename1=file_name[0]+".jpg"
    p2=os.path.join("./Segmentation",filename1)
    Gt=cv2.imread(p2,cv2.COLOR_RGBA2GRAY)
    gt2=cv2.resize(Gt, dim)
   
    
    fig, axs = plt.subplots(nrows=1,ncols=4, sharex=True, figsize=(12, 4))
   
    axs[0].set_title('Input Image').set_color('blue')
    axs[0].imshow(inp)
    axs[0].axis("off")

  

    axs[1].set_title('Ground Truth').set_color('blue')
    axs[1].imshow(gt2)
    axs[1].axis("off")

    axs[2].set_title('Model Prediction(ID)').set_color('blue')
    axs[2].imshow(segmented_image)
    axs[2].axis("off")

    axs[3].set_title('Model Prediction(ID)-Canny').set_color('blue')
    axs[3].imshow(feature_segmented_image)
    axs[3].axis("off")
    fig.savefig(p ,bbox_inches='tight', pad_inches=0.1 , facecolor='white',edgecolor="blue")
    return render_template("display.html", user_image = p)

if __name__ == "__main__":
    app.run(debug=True)

    

