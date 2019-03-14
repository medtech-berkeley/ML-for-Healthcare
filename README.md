# ML/Data Science for Healthcare

## What is Machine Learning?

Machine learning is when a computer has been taught to recognize patterns by providing it with data and an algorithm to help understand that data.

## Why are we doing this?

Machine Learning and Data Science is becoming more prevalent in healthcare.

![](https://lh4.googleusercontent.com/KcYpZhOGTDjlNXj0jA1T1xnOaMqjnDCY-YvZ6eaGcXSSTeVCIpC04qHmQmGbQwyu22stLSHQs-iRFNX5sXO7M8KAj1F-8bC_mw2-jXtg_j7syXnnPGQP8IeWBXTR29__qqPWxhZZzH8)

![](https://lh6.googleusercontent.com/BnKIhv2lauWKMN_d7bfneFixArXEDmZDxLYj_3qAgxUmJo0SJWwIWo8tSMmzKKiJMfXXXT9GZ8mZwd_QFE0VChkTkk9j_F-pHucKyKdZn0KFlVihfWvm3TjUwwl_4wZ2O-fL1lQ5Lh8)

![](https://lh5.googleusercontent.com/sc_43GUQldSM_2j5hvUhUcO69m82_QtLvBhCTHv2tcsCxmnEeqXRSPFeuofy2eFLr_OwBV9E3V2QoM9wZQWiaJEtDJdUTlHhRsoYOeH_cA_S7lPcUmCfxqAJVDvekZKU5PU6xnDJEEM)

## Types of Problems
1. Classification
  - The purpose of the Classification model is to determine a label or category – it is either one thing or another. We train the model using a set of labelled data. 
2. Regression
  - A Regression model is created when we want to find out a number – for example how many days before a patient discharged from hospital with a chronic condition such as diabetes will return.
3. Clustering
  - We would create a Clustering model if we had a whole bunch of data but we didn’t have a determined outcome, we just wanted to see if there were any distinctive patterns.


## Classification

**GOAL**: to create an algorithm that draws a line between the two labeled groups, called a ***decision boundary***.



![](https://lh4.googleusercontent.com/4m0DF7np9Ba0Fok9HCu_0bFrU4yIohDlpeNNyaR_Qnoa25wDFp2WyfVxet4hZjrTf6rH4re09xjczgG5zJ8MzFxe1TdsRwWWsXAJMF4kRF1gatkXL1GRpCp0FPvghq3zYEwcbXGKn4E)


Note: the ‘x’s on the graph above represent a data point, here a two-dimensional point (x,y) where x refers to the item’s color and y the item’s size. 

In general, data points will have some *n* amount of features (here, n = 2), and will thus lie on some *n-*dimensional space. 

When we are classifying data, we want to find some boundary that divides our space in two regions: one where, say, all data points that are ‘apples’ live, and the other region is where all data points that are ‘oranges’ live. For now, we will only consider the case where we have two classes two separate, but we can extend this to any amount of classes!


## How to actually do this…

In general, our model needs to have 2 things:


1. a ***predictor*** function: $f$, that maps input data to a predicted label
2. a ***loss*** function: $L$, that maps a predicted label to a loss value

Once we have these two things defined for us, we can ***optimize*** our predictor function so that it predicts a label as well as possible.  In other words, we want to ***minimize*** our loss.  

In order to find the minimum of our loss function, we’ll use an algorithm called **gradient descent**.



## Methods for Classification

Imagine we our problem is something more realistic than classifying fruits as apples and oranges. Let’s say we want to predict if Arjun will wear a Patagonia for based on today’s forecast.

This could be as easy as just finding a threshold temperature and claiming if the input temperature is below a certain threshold (say, 70˚F), we will output 1 (yes). If the temperature is above the threshold, we will output a 0 (no).

![](https://d2mxuefqeaa7sj.cloudfront.net/s_B573EEDBADB84059279FDEBC36E02F1848CF93650EE1B337E3406E49DD7BB676_1552526680460_image.png)



## Logistic Regression

However, life is not as black and white as the above model would suggest.  

In reality, Arjun won’t immediately put on a Patagonia the moment it drops below some predefined temperature. It’s more as if at any temperature he has a certain “chance” of putting on a Patagonia. Maybe at 45 F he would have a 95% chance of putting on a Patagonia, and at 60 F he would have a 30% chance of putting on a Patagonia.

To better model this, we use ***logistic regression*** to find these probabilities. This involves fitting a logistic curve (like the one below) to our data. To do this, we again use **gradient descent** to choose the best parameters for the model.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_B573EEDBADB84059279FDEBC36E02F1848CF93650EE1B337E3406E49DD7BB676_1552527081334_image.png)


The general form of the logistic model is:


$$\displaystyle\hat{y} = f(x) = \frac{1}{1 + e^{-w^Tx}} \in [0,1]$$ 

which represents the probability that $x$ results in a yes.  Our loss function is ***Cross Entropy Loss***:


$$\displaystyle L = -\sum_{i=1}^ny_i\log(\hat{y_i}) - (1 - y_i)\log(1 - \hat{y_i})$$

(we will get to the reasoning another day).

So then we can take the gradient of the loss function, and perform **gradient descent** to find the best set of parameters to help us understand when Arjun will wear his Patagonia.

(I realize we haven’t actually gone over the gradient descent algorithm at all, we’ll save that for next week don’t worry)


## Since we haven’t really delved into the details of how to train a model…

Let’s go with the simplest model that requires NO training!!!


## K-Nearest Neighbors Classifier

**The algorithm:**
Preprocessing: Split data up into 2 main groups:

- training set
- test set

Training time: do nothing. 
Test time: you are given an unlabeled point (from the test set). To predict its label, look at the labels of the ***k nearest training points*** (in the training set) and make an estimate.

- Categorical data: majority vote on the class.
- Numerical data: take an average. Perhaps weighted by distance.

**What does is mean to be “near”?**
The K-Nearest Neighbors algorithm generally uses ***euclidean distance*** to quantify how near or far two points in the data are from each other.

Let $x\in\mathbb{R}^n$ be one data point and $x'\in\mathbb{R}^n$ be another data point (*note: these are both vectors*). The ***euclidean distance*** between these two points would be:


$$\displaystyle d(x, x') = \sqrt{\sum_{i=1}^n(x_i - x'_i)^2}$$

**What is K??**
K is a ***hyperparameter***. You choose it. Here's a nice visualization of the effect of K on classification.

- If K = 1, classify point based on only 1 nearest. Moving a little can cause you to ip classes (high variance). But each training point classified correctly (low bias).
- If K = N (# training points), every point gets the same classification as it looks at entire training set. Low variance. But you will surely get many things wrong (high bias).

**So… how do we determine the best value for K?**

Solution: try a bunch of them…

![](https://d2mxuefqeaa7sj.cloudfront.net/s_B573EEDBADB84059279FDEBC36E02F1848CF93650EE1B337E3406E49DD7BB676_1552529337927_image.png)




## Now to do it for ourselves!!

We are going to tackle the task of predicting whether or not someone has diabetes. 

We will be using the diabetes data set which originated from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php).

Go to https://github.com/medtech-berkeley/ML-for-Healthcare and clone the repository.  

Make sure you have Python and Jupyter installed on your computer.


    python3 -m pip install --upgrade pip
    python3 -m pip install jupyter

Once you have cloned the repository, navigate to it inside your terminal and run: 


    pip3 install -r requirements.txt

Now you should be good to go!!

(if this part is proving to be difficult then you can download Python through Anaconda and that should take care of everything)

