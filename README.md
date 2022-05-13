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




