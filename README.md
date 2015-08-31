# Face Alignment at 3000fps
It is an implementation of Face Alignment at 3000fps via Local Binary Features, a paper on CVPR 2014

# Interpret Paper 
If you are a Chinese, you can go to my blog for more details. [link](http://freesouls.github.io/2015/08/30/caffe-implement-l2-normlization-layer/)

# License
If you use my work, please cite my name (Binbin Xu), Thanks in advance.
This project is released under the BSD 2-Clause license.

#How To Use
####Requirements:
1. OpenCV(I just use the basic structures of OpenCV, like cv::Mat, cv::Point)
2. cmake

####Prepare: 
1. you should change some image PATH in main.cpp and utils.cpp(function LoadImages) for correctly running the program.
2. set appropriate parameters in Train() (in the file of main.cpp)

####Compile:
```
mkdir release
cp CMakeList.txt ./release
cd release
cmake .
make
./application train ModelName # when training
./application test ModelName # when testing 
./application test ModelName imageName # when testing one image
```

# Notes
- The paper claims for 3000fps for 51 landmarks and high frame rates for different parameters, while my implementation can achieve several hundreds frame rates. What you should be AWARE of is that we both just CALCULATE the time for predicting the landmarks, EXCLUDES the time that detecting faces.
- If you want to use for realtime videos, using OpenCV's face detection algorithm will achieve about 15fps, since 80% (even more time is used to get the bounding boxes of the faces in an image), so the bottleneck is the speed of face detection, not landmarks predicting. You are required to find a fast face detector(For example, [libfacedetection](https://github.com/ShiqiYu/libfacedetection))
- I use the opencv face detection method, you can change to what you like, as long as using the same face detector in training and testing
- it can both run under Windows and Unix-like systems.
- it can reach 100~200 fps(even 300fps+, depending on the model) on a single i7 core when the model is 5 or 6 layers deep, predicting 68 landmarks. The speed will much faster when you reduce the 68 landmarks to 29, since it uses less parameters.
- for a 68 landmarks model, the trained model file(storing all the parameters) will be around 150M, while it is 40M for a 29 landmarks model. 
- the results of the model is acceptable for me, deeper and larger random forest will results better results, but with lower speed. 


# Results Image:
###1. detect the face
![](./detect.png)
###2. first use the mean shape for initial shape:
![](./initial.png)
###3. predict the face landmarks by refine the initial shape
![](./final.png)


# Future Development
- I have add up the openMP to use multithread for faster training, it is really fast, takes an hour when the model is 5 layers deep and 10 trees in each forest with about 8000+ augmented images.
- I have already develop the multithread one, but the time for predicting one image is slower than sequential one, since creating and destroying threads cost more time.
- I will optimize it and update it later.
- Second, I will also develop a version on GPU, and will also upload later.

# THANKS and More
Many thanks goes to those appreciate my work.

if you have any question, contact me at declanxu@gmail.com or declanxu@126.com, THANKS.
