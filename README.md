# face alignment at 3000fps
It is an implementation of Face Alignment at 3000fps via Local Binary Features, a paper on CVPR 2014
#How To Use
```
cd release
cmake .
make
./application train ModelName # when training
./application test ModelName # when testing 
./application test ModelName imageName # when testing one image
```
# Notes
    you should change some image path for correctly running the program.
    it can both run under Windows and Unix-like systems.
    it can reach 100~200 fps(even faster) on a single i7 core when the model is 5 layers deep,
    it can reach 50~60 fps on a single i7 core when the model is 6 layers deep with 10 trees in each Random Forest.


# Future Development
    I have already develop the multithread one, but the time for predicting one image is slower than sequential one, since creating and destroying threads cost more time.
    I will optimize it and update it later.
    Second, I will also develop a version on GPU, and will also upload later.

if you have any question, contact me at declanxu@gmail.com or declanxu@126.com, THANKS.
