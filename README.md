# Review_Sentiment_Classification
## Introduction 

This is a project for binary sentiment classification of movie reviews.

use main.py to run main code and test a model in terminal

or use manage.py to input your own reviews on local host


### Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
## Process:

-Load source .txt files and turn them into a .csv dataframe for for further convenience;

-Use TF-IDF Vectorizer to make a matrix of vector features;

-Split reviews to test and train sets;

-Use a classification model to train on data (Logistic Regression, Naive Bayes or K-Nearest Neighbours);

-Compare classifier's accuracy using 6 different methods (ROC AUC Score, F1-Score, Accuracy Score, Confusion matrix, mean accuracy (.score method), classification report);

-Try out manual input of review in terminal (or run django server using manage.py and enter your review there). 
