# Emotion-Detection-2.0

Analyze audio and video to predict emotions.

## Team Members:
#### 1)Akriti Jain 
#### 2)Navya Mamoria 
#### 3)Aayush Kumar 
#### 4)Suriya R S 
#### 5)Chaitanya Gupta 

## Introduction

Emotion recognition has become an important topic of research in recent years due to the multiple areas where it can be applied, such as in healthcare or in road safety systems etc. Human emotions can be detected using speech signals, facial expressions and body language. Thus, an algorithm that performs detection, extraction, and evaluation of these features will allow for automatic recognition of human emotion in images and videos.
In the last five years, the field of AI has made major progress in almost all its standard sub-areas, including vision, speech recognition and generation, image and video generation coupled with advancement in various deep learning techniques, now it is possible to predict human emotions with much higher accuracy.


### Prerequisites

-Knowledge of various deep learning techniques like CNN, LSTM etc. <br />
-Some insight into frameworks such as OpenCV, Librosa, Keras, Tensorflow etc and python libraries like numpy, pandas, sklearn,etc.


### How AI-Emotion Analysis works

On a high level, an AI emotion application system includes the following steps:<br />

Step #1: Acquire the image frame from a camera feed or extract the audio from .wav file. <br />
Step #2: Preprocessing of the image/audio file (cropping, resizing, color correction, normalisation).<br />
Step #3: Extract the important features with a suitable model<br />
Step #4: Perform emotion classificati0n


### Our Approach

Initially, we were considering multimodal emotion detection method based on the fusion of speech and facial expression. According to the processing of different modal signals in different stages, it can be divided into signal-level fusion, feature-level fusion, decision-level fusion, and hybrid fusion. In this paper, the decision-level fusion method is used to independently inspect and classify the features of each modal and merge the results into a decision vector.
![](https://www.mdpi.com/sensors/sensors-21-07665/article_deploy/html/images/sensors-21-07665-g001.png)
![](https://static-01.hindawi.com/articles/wcmc/volume-2021/6971100/figures/6971100.fig.001.svgz)
<br />However, this approach is highly experimental as researchers are still working in this area to get decent accuracy, therefore on advice from our mentors, we dropped this idea. Finally, we decided to train two independent models and combine their result in the end to predict the emotion based on a live video feed wherein the video will be framed at intervals using OpenCV and the audio will be extracted from the video and used as the speech input.
Due to the autonomous nature of both the models, two of us (Akriti and Navya) worked on the audio part and the others (Aayush, Chaitanya and Suriya) worked on the video part. 


### Visual Analysis

### DataSet Used
## FER 2013
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.


### Models Used
First we tried using normal convolutional layers with BatchNorm ,MaxPooling and dropout in between . We used standard filter size of 3 by 3 , padding as 'same' and activation function as relu . For top layers we added a few DenseLayers ,finishing with a 7 layer classifier . As we were able to only achive a 58 % val_accuracy through this model we tried various otherthings. 
## *Model 1*
```python
#Build Model
model = models.Sequential()

model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, data_format='channels_last'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

 model.summary()
```
## *Model 2*
In this model we are using a well known and well tested architecture made of inception modules with residual links i.e InceptionResnetV2 as our model. Rest of the concepts used are same as the previous one.

```python
InceptionResNetV2=tf.keras.applications.InceptionResNetV2(weights='imagenet',input_shape=input_shape, include_top=False)
model = models.Sequential()
# load the model
model.add(InceptionResNetV2)

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.75))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.50))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()
```
## InceptionResNetV2 Architecture




