import numpy as np 
import pandas as pd
from PIL import Image
from flask import Flask,render_template,request
import cv2
import tensorflow
import io
from keras.preprocessing import image
import base64


app=Flask(__name__)
Model=tensorflow.keras.models.load_model("Braintumor.keras")


@app.route("/")
def index():
    return(render_template('index.html'))
@app.route('/predict',methods=['GET','POST'])
def predict():
    img = request.files['image']
    image=Image.open(img)
    resize_image=image.resize((256,256))
    final_image=np.expand_dims(resize_image,axis=0)
    prediction=Model.predict(final_image)
    class_name=['Glioma', 'Healthy', 'Meningioma', 'Pituitary']
    output=class_name[np.argmax(prediction)]
    img_io = io.BytesIO() 
    image.save(img_io, 'PNG') 
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
    img_data = f"data:image/png;base64,{img_base64}"

        
    return render_template('index.html', prediction=output, image_data=img_data)


if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000)   
    


    

    





 



