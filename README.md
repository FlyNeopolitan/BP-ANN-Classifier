# Classifier
Hi everyone! It's a Classifier based on ANN and BP implemented with C++.
## Is it easy to use?
Yes! You don't have to worry about implementations. You just need to convert your input and output to vectors!
## How about the accuracy?
The accuracy for classifying Iris is 100%. (Data set from UCI Machine Learning Repository)  
Also, it passes all basic tests.
## How do I use them?
1.Use std::initializer_list to create a Classifier  
2.Use data set to train the Classifier  
3.Classify!  
For details, check Classifier/ann_classifier.h and Classifier/classifier.h; If you are still confused, check main.cpp for a sample for Iris data.
## I am still CONFUSED!
Here is another sample:  
Assume you want to classify a 2x2 image, which has 4 pixels and each pixel has 2 state 0(black) or 1(white). 
Thus, we can convert the image to a vector: if an image is  
black black  
white white  
Then we convert it to a vector {0,0,1,1}  
Assume that we want to classify two kinds of images, so when the output vector is {1,0}, it indicates the image1, and when ouput is {0,1}, it indicates the image2.  
So first we read in data from image, and convert it to vectors.  
Then, we train the Classifier with data set in vector format.   
At last, we could use Classifier to classify the input!  


