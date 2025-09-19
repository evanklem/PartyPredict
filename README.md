# PartyPredict

Naive Bayes survey-based classifier that predicts political party affiliation from responses before the survey ends.

## Overview
PartyPredict is a Java application that collects survey responses, stores them in CSV format, and trains a Naive Bayes classifier to infer political party. It evaluates predictions with accuracy, precision, recall, and F1 scores, and improves as new data is added.

## What I Learned
- Designing clear, unbiased survey questions.  
- Encoding categorical answers and managing CSV data storage.  
- Implementing and tuning a Naive Bayes classifier in Java.  
- Applying Laplace smoothing and weighted features to stabilize results.  
- Measuring performance with standard classification metrics.  

## Challenges Overcome
- Handling conditional independence limitations in Naive Bayes.  
- Preventing zero-probability issues with rare responses.  
- Balancing classes to avoid skewed metrics.  
- Making the system retrainable as new labeled data arrives.  
