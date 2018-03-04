# Custom-object-detection-using-TensorFlow-API

Introduction


In this project I am trying to accomplish the task of detecting objects in an image/video using TensorFlow and OpenCV.
Object detection is not as simple as object classification. An object classification algorithm tries to categorize entire images into different classes whereas an object detection algorithm tries to identify the objects in an image and draw bounding boxes around them. We need much more powerful algorithm for object detection in the form of deep neural networks(R-CNN,Faster R-CNN etc).

One of the core challenges of machine learning and computer vision has been to identify and detect multiple objects accurately in a single image.In this project I am using the opensource TensorFlow Object detection framework built on top of TensorFlow. The TensorFlow model can detect many common objects with a good enough accuracy but it has the following limitations as well:
•	Some objects are detected and labeled wrongly.
•	There are not many objects that the model detects.

I have trained the model on custom objects(soccer ball etc) and also extended the API to detect objects in a video game(GTA 5).

Implementation details
The TensorFlow object detection API  has many dependencies that need to be included for the API to work on.
There are two versions of tensorflow that the API can run on:
1.	Tensorflow-cpu
2.	Tensorflow-gpu
Training on the tensorflow-gpu is much faster and can train large batches at once.It needs a powerful CUDA-enabled NVIDIA GPU (GTX 850m series or greater) 
Lets take a look at the concepts one by one:
1.	Tensorflow: Tensorflow is an open source software library that uses data flow graphs for numerical computation.Nodes in the graph represent mathematical operations and edges represent tensors.Tensors are n-dimensional arrays that is operated on by the nodes of the graph.
2.	CUDA: A parallel computing platform developed by NVIDIA that allows users to use the capabilities of their powerful GPU’s for deep learning.It gives direct access to the GPU’s virtual instruction set for the execution of compute kernels.


Now lets take a look at the concepts I implemented one by one:
1.	Object detection in custom images using the Tensorflow API
First, I used the Tensorflow API to detect objects in custom images.As I stated earlier,the API itself has several dependencies that need to be installed first.
•	Python 3.6
•	Numpy
•	Matplotlib
•	Jupyter Notebook
I used Anaconda3 as it comes with Python3 and all its dependencies installed.After downloading the API,I downloaded the protoc file which is used to run the protocol buffers. The protocol buffers are used to generate classes in C,python that can load,save data in an easy way. Used the ssd_mobilenet which is trained on 90 different classes i.e it has the capacity to detect 90 different classes.
 
 

2.Extending the API to detect objects in a video
To use the API to detect objects in a  video we can either load in a video or open the webcam and detect objects in real-time.Instead of plotting the output to the jupyter notebook we activate the fron camera using cv2.VideoCapture().
Inside the TF session I am running the object detection algorithm for each frame returned by the video by adding the line while True: ret,img_np=cv2.Videocapture().
I display the video using cv2.imshow()


3.Training a custom object on the model
The TF model can detect about 90 different classes of images.In this part I train the model on a custom class(other than the 90 different classes).This part was not as easy as just adding another class to the model and making the model detect it.
Here are the following steps that I implemented:
Step 1:Collecting the images that I want the model to detect(soccer ball in this case)
I try to collect soccer ball images of different sizes (preferably around 150-200) in the .jpg format and stored them in a folder
Step2:Hand labeling all the images using the labelImg annotation tool.
In this step I select each image(from the 200 images) and draw bounding boxes around the soccer ball in each image.

 This annotation tool automatically generates a xml file for each image which is then converted into a csv format using .This is necessary because we need to convert the image files to the PASCAL VOC dataset format which in turn can be conveniently converted to TF records which the TF model accepts as input. 

We split the entire dataset of images into train and test folders(90% train and 10% test).Along with the record files train.record and test.record we give the train and test images as input to the TF model.
We set the configuration file and allow the model to run. 

 After the loss was reduced to less than 0.5 and allowing the model to run for about 10k steps I stopped the model. The model outputted a frozen inference graph(.pb file) , a checkpoint file and a .meta file for every 1000 steps
 
 

4.Detecting the objects in a video game
I was also able to complete this step . In this step I am using the object detection API to detect the objects in a video game (any realistic video game would work).
In order to make this work I had to write a code to grab my screen into a terminal where the TF API would be implemented and it would identify the objects.


  
References

1.https://github.com/tensorflow/models/tree/master/research/object_detection

2.https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

3.https://www.tensorflow.org/get_started/

4.https://pypi.python.org/pypi/grab-screen



