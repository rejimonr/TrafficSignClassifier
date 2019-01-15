# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!. All code is available in Traffic_Sign_Classifier.ipynb notebook and the final outputs are available in the Traffic_Sign_Classifier.html.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

Simple display of an random image was done initally. Expanded to include a snapshot of 8 images. 
Also included three bar charts indicating the distribution of the sample images across the classes in the three sets. This was done in the end and does indicate what I had come to suspect after the results and reading various articles online that data augumentation should have been done to increase the accuracy. 

*Visualization is available in the html file. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


Started with two very basic preprocessing steps - Grayscaling and normalization - based on the LeNet lesson suggestions and need to have an 32x32x1 input. 
Decided to go with the simple normalization based on the suggestion in the paper in the project. (On a side note, when the accuracy was very low initially from the model I suspected some issue with this step. Checked the mean to ensure that normalization did bring down the mean to close to zero. The low accuracy wa due to a bug in the implementation of the model code!)

I should have gone ahead with the data augmenation of the samples (based on the accuracy results and the uneven distribution of the data across the classes). Will do this after this submission. 

Another area which I want to explore in furture is the use of YUV color space as suggested in the paper. 

*Visualization is available in the html file. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


  1. 5x5 convolution (32x32x1 in, 28x28x6 out)
  2. ReLU
  3. 2x2 max pool (28x28x6 in, 14x14x6 out)
  4. 5x5 convolution (14x14x6 in, 10x10x16 out)
  5. ReLU
  6. 2x2 max pool (10x10x16 in, 5x5x16 out)
  7. 5x5 convolution (5x5x6 in, 1x1x400 out)
  8. ReLu
  9. Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
  10.Concatenate flattened layers to a single size-800 layer
  11.Dropout layer
  12.Fully connected layer (800 in, 43 out)



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I started with the standard LeNet implementation originally and hence carried over the Adamoptimizer over to my final model. The training was primarily done by modifying the number of epochs, batch size, learning rate and the keep probability for the dropout layer.
Multiple trials were done to come up with the final values of the hyperparameters. (More work needs to happen here. This is not the best case obviously)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.944
* test set accuracy of 0.928

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
 
     I started with the standard LeNet implementation from the classes because that was a very solid base with proven results available and I assumed only a finetuning of the hyperparameters will needed. After fixing all the bugs in my implementation code I was able to run few training iterations to see what is happening with the final validation accuracy results. Spent considerable amount of time trying to fine tune the learning rate, batch size and epoch numbers to increase the validation accuracy to above 0.93. Increased the keep prob to 0.6 from 0.5 finally to achieve the required validation accuracy with the below parameters. (This reduced the overfitting which was happening)
     
    Epoch = 70
    Batch = 150
    rate = 0.00099
    keep_prob = 0.6
    
   After further research decided to go with the modified model to further increase the accuracy of the model. The primary focus was (based on several online discussions and suggestions from the project paper) to incorporate the features detected in the earlier convolutions also into the final fully connected layer. This resulted in immediate improvement and further finetuning was done on the hyper parameters to get to the below final values. 
    Epoch = 70
    Batch = 128
    rate = 0.00099
    keep_prob = 0.6
   


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I picked up 10 images from the web (some duplicate) to test it out. The images were much better than the sample ones (as I was initially doubtful on the model created). The mode was able to identify the images very easily as they were bright and had very little noise or disturbances. 

*Visualization is available in the html file. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

As explained above the simple images were detected easily by the model. I will need to try out with some more difficult images after this iteration of the submission. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Again as expected for the simple images it was able to detect all of them accurately (accuracy of 1.0). Visualization was done to see what is happening with the top 5 probabalities. 

##### Future iteration TODOs
Data Augmentation (This should improve the accuracy more)
Use of YUV color space
Exprimenation with other combination of layers and activation functions
Test out the models with more difficult new images (more numbers as well)
Implement the optional visualization task to see how the different layers are identifying the different features. 