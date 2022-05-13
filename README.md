# Emotion-Detection-2.0

Analyze audio and video to predict emotions.

## Team Members
* [Akriti Jain ](https://github.com/akriti-github)
* [Navya Mamoria ](https://github.com/navyaa2002)
* [Aayush Kumar ](https://github.com/aayushk68)
* [Suriya R S](https://github.com/suriya030)
* [Chaitanya Gupta](https://github.com/code-x-1318)

## Introduction

Emotion recognition has become an important topic of research in recent years due to the multiple areas where it can be applied, such as in healthcare or in road safety systems etc. Human emotions can be detected using speech signals, facial expressions and body language. Thus, an algorithm that performs detection, extraction, and evaluation of these features will allow for automatic recognition of human emotion in images and videos.
In the last five years, the field of AI has made major progress in almost all its standard sub-areas, including vision, speech recognition and generation, image and video generation coupled with advancement in various deep learning techniques, now it is possible to predict human emotions with much higher accuracy.
## Scripts
* ### Visual_FinalModel.ipynb  This script contains model for emotion classification.
* ### RealTimeEmotionDetection.ipynb  This script contains the code to extract video and predict live emotions .
* ### Audio_final.ipynb ###


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


## VISUAL ANALYSIS

### Our WorkFlow
* Segregating the FER2013 dataset into Train,Test and Validation Dataset
* Designing a CNN based Network for classifying emotions .
* Training the Network using the Train_dataset.
* Obtaining the Results as acc vs epoch and loss vs epoch
* Testing the trained model in RealTime.  

### FER 2013
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.





### *Model 1*
First we tried using normal convolutional layers with BatchNorm ,MaxPooling and dropout in between . We used standard filter size of 3 by 3 , padding as 'same' and activation function as relu . For top layers we added a few DenseLayers ,finishing with a 7 layer classifier . As we were able to only achive a 58 % val_accuracy through this model we tried various otherthings.  
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
Model 1
  
  Train acc 70 percent
  Val acc   58 percent

As the validation acc is quite low we went for different Network architecture ,one of them is shown below.


We have tried also VGG16 model. Since , FER2013 dataset contains only grayscale images , but our VGG16 model accepts only 3 channel images. So, what we did here is modify the 3 Color channel input to 1 channel input and we modified the kernels used on the first layer of 3 channel size to 1 channel by taking average on all corresponding pixels of RGB channels.In addition , we used pretrained weights (transfer learning) from ImageNet Dataset. Since , we didnt get satisfoctory result we 
have gone with below model.

### *Model 2*
In this model we are using a well known and well tested architecture InceptionResnetV2 as our BaseModel. Rest of the concepts used are same as the previous one.

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


### InceptionResNetV2 Architecture

![](https://www.researchgate.net/profile/Masoud-Mahdianpari/publication/326421398/figure/fig9/AS:649354851385344@1531829669740/Schematic-diagram-of-InceptionResNetV2-model-compressed-view.png)

Inspired from the performance of Inception and ResNet ,the idea of residual links have been introduced into the inception module from InceptionV4 . Based on that 
3 different InceptionResnet modules were formed namely A,B and C .
![](https://miro.medium.com/max/1400/1*WyqyCKA4mP1jsl8H4eHrjg.jpeg)
*(From left) Inception modules A,B,C in an Inception ResNet. Note how the pooling layer was replaced by the residual connection, and also the additional 1x1 convolution before addition. (Source: Inception v4)*

The idea of using multiple convolutional filters and MaxPooling at the sametime towards a featuremap and then combining the results from those multiple operations to form a new feature map forms an inception module(InceptionV1) , doing some minor changes to this to make training faster we will arrive at InceptionV4.
ResNet-50 is a Network consists of normal convolutional layers with Residual link . what residual link does is feeds the output of a particular layer to input of next to next layer .
Both of these ideas were used to create InceptionResNetV2.

### Results
We were able to arrive at a result of (MODEL 2) ,
                              
                              Train accuracy 77   percent
                              val   accuracy 69   percent
                              test  accuracy 68.8 percent
                              

 ![image](https://user-images.githubusercontent.com/95056319/168308963-2dda5e2f-7cc7-4ead-9c75-06e31780cd38.png)
  
![image](https://user-images.githubusercontent.com/94068599/168277692-d8fcfd11-fab2-43c2-a511-71e074d346b1.png)

*The BenchMark accuracy on FER2013 dataset till date is 76.82 percent which done by ensemble ResmaskingNet with 6 other CNNs.* 



## Audio Analysis
 
### Workflow
 
Dataset- RAVDESS and TESS
Software used- Google collab
Framework-  Sci-kit learn, librosa, and Keras
Algorithm- 1D CNN and/or LSTM
Using TensorFlow backend.

#### Step 1- Loading the dataset.
  
Firstly, we mounted the datasets on Google drive and unzipped them separately.

#### Step 2- Testing audio files 

Tested one audio file by plotting its waveform and spectrogram.

#### Step 3- Feature Extraction 

The next step involves extracting the features from the audio files, which will help in identifying the components of the audio signal that can easily help to distinguish emotions embedded. For feature extraction, we use the LibROSA library in python, one of the libraries used for audio analysis.

Features extracted- MFCC(Mel-Frequency Cepstral Coefficients) AND Mel spectrogram. 
 
 #### Step 4- Creating a dataframe

Creating a final data frame consisting of features and emotions.

#### Step-5   Splitting the data into test and train sets.

#### Step-6 Data Preprocessing 

Using MinMaxScaler and label encoding emotions of y train and y test set.
#### Step-7 Building 1D CNN model

We built A 1D CNN model with 12 layers-used ReLu function as our activation function for 4 NN layers and a final layer of NN as a softmax classifier. We also added three dropout layers to prevent overfitting.
 
#### Step-8 Prediction

Predicting the emotions of the Y test.

#### Step-9 Confusion Matrix

For checking the model's accuracy and finding the number of true positives, true negatives, false positives, and false negatives.

### Conclusion

-The 1D CNN model gave an F1 score of 81% for testing seven emotions that are happy, sad, neutral, disgust, angry, fear, and surprise.
-Other technologies like LSTM takes a lot more time for training and testing.

### Audio_final.ipynb is the file that contains the code for audio analysis of the emotions.



## References

* https://arxiv.org/abs/2004.11823
* https://www.kaggle.com/datasets/msambare/fer2013
* https://arxiv.org/abs/1602.07261v2


