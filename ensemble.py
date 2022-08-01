from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import numpy as np
from skimage import transform
from sklearn import metrics
import statistics

import matplotlib.pyplot as plt


base_dir = r'D:\Anki\Python Projects\Workspace\NUS project\dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_fresh_dir = os.path.join(train_dir, 'Fresh')  
train_rotten_dir = os.path.join(train_dir, 'Rotten') 

test_fresh_dir = os.path.join(test_dir, 'Fresh')  
test_rotten_dir = os.path.join(test_dir, 'Rotten') 

image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

BATCH_SIZE = 32
IMG_SHAPE = 150

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='categorical')
augmented_img = [train_data_gen[0][0][0] for i in range(5)]

image_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=test_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='categorical')
test_data_gen_mod1 = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=test_dir,
                                                 target_size=(200, 200),
                                                 class_mode='categorical')


model1 = tf.keras.models.load_model(r'assets\first.h5')
model2 = tf.keras.models.load_model(r'assets\second.h5')
model3 = tf.keras.models.load_model(r'midnyt1.h5')
models = [model1,model2,model3]

def load(filename,shape=150):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (shape, shape, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('test.jpg',150)
print(model3.predict(image))
pred=[]
for model in models:
    if model==model1:
        pred.append(np.argmax(model.predict(test_data_gen_mod1)))
    elif model == model2:
        pred.append(np.argmax(model.predict(test_data_gen)))
    else:
        pred.append(np.argmax(model.predict(test_data_gen)))


ind = statistics.mode(pred)
print(ind)

# preds = [model.predict(image) for model in models]
# print(preds)
# preds=np.array(pred)
# summed = np.sum(preds, axis=1)

# # argmax across classes
# ensemble_prediction = np.argmax(summed)

print(test_data_gen.class_indices.keys())

validation_x=[]
for i in range( test_data_gen.__len__() ):
    validation_x.extend(
        test_data_gen.__getitem__( i )[1] 
        )
validation_x = np.argmax(validation_x,axis=1)
prediction1 = np.argmax(model1.predict(test_data_gen_mod1),axis=1)
prediction2 = np.argmax(model2.predict(test_data_gen),axis=1)
prediction3 = np.argmax(model3.predict(test_data_gen),axis=1)


confusion_matrix = metrics.confusion_matrix(validation_x, prediction1)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Fresh Apples','Fresh Banana','Fresh Oranges','Rotten Apples','Rotten Banana','Rotten Oranges'])
cm_display.plot()
plt.show()
confusion_matrix = metrics.confusion_matrix(validation_x, prediction2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Fresh Apples','Fresh Banana','Fresh Oranges','Rotten Apples','Rotten Banana','Rotten Oranges'])
cm_display.plot()
plt.show()
confusion_matrix = metrics.confusion_matrix(validation_x, prediction3)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Fresh Apples','Fresh Banana','Fresh Oranges','Rotten Apples','Rotten Banana','Rotten Oranges'])
cm_display.plot()
plt.show()

accuracy1 = accuracy_score(validation_x, prediction1)
accuracy2 = accuracy_score(validation_x, prediction2)
accuracy3 = accuracy_score(validation_x, prediction3)
# ensemble_accuracy = accuracy_score(validation_x, ensemble_prediction)

Precision = metrics.precision_score(validation_x, prediction2,average ='micro')
print("Precision Score of model2 = ",Precision)

print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2)
print('Accuracy Score for model3 = ', accuracy3)
# print('Accuracy Score for average ensemble = ', ensemble_accuracy)