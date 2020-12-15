# Predicting the Customer Sentiments from Yelp Dataset
Performing the sentiment analysis and building classification models to segregate the reviews as Positive or Negative.

## Summary
* [Introduction & General Information](#introduction--general-information)
* [Objectives](#objectives)
* [Data Used](#data-used)
* [Approach & Methodology](#approach--methodology)
* [Run Locally](#run-locally)
* [Conclusion](#conclusion)

## Introduction & General Information
- Sentiment Analysis is a process of determining whether a text is expressing positive, negative or neutral opinion about a particular product or services. Every business relies on the customer's requirements and feedbacks, so understanding what the customers feel about their services is a very crutial task.
- It consists of natural language processing, statistical and text analysis techniques to identify the sentiments of various words/phrases as positive, neutral or negative.
- Sentiment Analysis is carried out on the information gathered from social media, surveys, online reviews or other channels. Hence, most of the data obtained is unstructured which requires some pre-processing to be carried out.
- Natural Language Processing is a collection of methods to process the unstructured content and understand the text. In this project, we will use python's Natural Language Toolkit (NLTK) library to process and convert the customer reviews text into a machine readable form.

## Objectives
- Exploring the dataset containing information about various businesses and their customer ratings.
- Performing Sentiment Analysis of the customer reviews for Beauty and Spa Services located in the United States.
- Building Machine Learning models to predict whether the sentiment is positive or negative.

## Data Used
(Data Source: https://www.kaggle.com/yelp-dataset/yelp-dataset)
- Yelp dataset contains the information about the businesses from across 11 metropolitan areas.
- This dataset consists of JSON files for business information, reviews, customer checkins, tip details and user information.
- For the purpose of this project, business and reviews dataset has been filtered to obtain the data related to 'Beauty and Spas' business category.

## Approach & Methodology
- Convert the businesses and reviews JSON files to CSV. Load the data from these CSV files to pandas dataframe.

- Filter the data related to Beauty and Spa Services situated in the US from the business and reviews dataframe.

- Perform exploratory analysis on the filtered dataset.

- Categorize the ratings as 'Good Reviews' and 'Bad Reviews'. All the ratings are in the range 0 - 5, so categorize ratings above 4 as 'Good Reviews' and below 4 as 'Bad Reviews'.

- Pre-process the textual reviews by removing unwanted elements from the text before performing Natural Language Processing.

- Perform feature extraction for sentiment analysis using following techniques:
    - **Bag of Words Model:** Textual reviews are converted into tokens and each token is associated with a certain review label (Good or Bad)
    - **Polarity Scores:** Using VADER model for scanning the text and assigning positive, negative, or neutral polarity scores to the text.
    - **doc2vec Model:** Using Gensim modeling toolkit to convert text document to vectors to capture the relationship between different words.

- For the extracted features using each of the above feature extraction techniques, build and train sentiments classification models. Compute prediction accuracy to evaluate the model performance.

## Run Locally
- Make sure Python 3 is installed. Reference to install: [Download and Install Python 3](https://www.python.org/downloads/)
- Clone the project: `git clone https://github.com/setu-parekh/yelp-customer-review-sentiment-prediction.git`
- Route to the cloned project: `cd yelp-customer-review-sentiment-prediction`
- Install necessary packages: `pip install -r requirements.txt`
- Download the following data JSON files from [here](https://www.kaggle.com/yelp-dataset/yelp-dataset):
  - yelp_academic_dataset_review.json
  - yelp_academic_dataset_business.json
- Save these files in the route `cd yelp-customer-review-sentiment-prediction`.
- Run Jupyter Notebook: `jupyter notebook`
- Select the notebook to open: `customer_sentiment_analysis.ipynb`

## Conclusion
- Performance evaluation of the classification models based on the features extracted using following techniques:
  - **Bag of Words**:
    - Naive Bayes Model was trained using these features. It could predict the sentiments of a review with ~65% accuracy.
  - **VADER Polarity Scores**:
    - Random Forest Model was trained using these features. It could predict the sentiments of a review with ~82% accuracy.
  - **Word Embedding**:
    - Random Forest Model was trained using these features. It could predict the sentiments of a review with ~86% accuracy
    - Support Vectors Model was trained using these features. It could predict the sentiments of a review with ~87% accuracy

- Inferences:
  - Bag of Word technique is the most fundamental technique for feature extraction since it uses only the words as features for sentiment analysis. Classification model built on features extracted using this technique has the lowest prediction accuracy.

  - Word Embedding feature extraction technique is based on the principle that the most similar group of words will be placed together and dissimilar words will be placed far from each other. It can very well capture the relation between the words. Classification model built on features extracted using this technique has the highest prediction accuracy.
