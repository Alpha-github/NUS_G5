from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

labels = ['Fresh','Rotten']

@app.route('/')
def index():
	return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
	img = request.files['img']
	img.save("img.jpg")

	model1 = tf.keras.models.load_model(r'./first.h5')
	
	def load(filename,IMG_SHAPE=150):
		image = cv2.imread(filename)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (IMG_SHAPE,IMG_SHAPE))
		image = np.reshape(image, (1,IMG_SHAPE,IMG_SHAPE,3))
		return image

	pred = labels[np.argmax(model1.predict(load('img.jpg',200)))]
	print(pred)

	return render_template("prediction.html", data=pred)

if __name__ == "__main__":
	app.run(debug=True,host="0.0.0.0")