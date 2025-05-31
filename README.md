# Comparing NLP Models: Data Scale Impact on Accuracy and Training Speed

This project is based on the null hypothesis that the primary difference among various Natural Language Processing (NLP) models lies in their training speed and scalability when exposed to large volumes of data, rather than in their ultimate predictive accuracy. We start from the assumption that, given sufficiently large datasets, different NLP models will converge toward a similar maximum accuracy. Thus, the main distinction among them is their efficiency and scalability during training, rather than a fundamental difference in predictive power when exposed to large datasets.

To prove or disprove this hypothesis, the project will analyze the performance of 5 progressively powerful Models on 5 famous Kaggle datasets, by progressively increasing the volume of data used for training, and comparing the results.

## Table of Contents
- [Models](#models)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Usage and Installation](#usage-and-installation)
- [Results](#results)
- [References](#references)

## Models
The five Models chosen for this experiment were (#TODO COLOCAR OS LINKS E UMA DESCRIÇÃO CURTA DE CADA MODELO):
- Simple TFIDF BoW
- LDA
- GloVE
- BERT

## Datasets

The datasets chosen for this experiment were:
- [Sentiment140](https://www.kaggle.com/kazanova/sentiment140): A large collection of 1.6 million tweets labeled for sentiment (positive, negative, neutral), used for sentiment analysis on social media text.
- [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews):Contains over 560,000 food product reviews from Amazon with star ratings from 1 to 5, widely used for sentiment analysis and opinion mining.
- [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset): Contains ~200,000 news headlines labeled by category, used for multiclass topic classification.
- [DBpedia Ontology Classification](https://www.kaggle.com/dbpedia/ontology-classification): Over 560,000 Wikipedia article abstracts categorized into 14 ontology classes for large-scale multi-class text classification.
- [AG News Topic Classification](https://www.kaggle.com/agnews/ag-news): 120,000 news articles labeled into 4 topics (World, Sports, Business, Sci/Tech) for topic classification benchmarking.

## Methodology

To test the null hypothesis, the following methodoloy was used:
- Train and Test all five models with progressively more data (#TODO COLOCAR A PROGRESSÃO PERCENTUAL DOS DADOS USADOS), always using the same training and testing data for all five models, for all datasets.
- Comparing and plotting the accuracy results for each model against each other.

## Usage and Installation
(#TODO Sessão curta de instalação e etc)

## Results
(#TODO Sessão de resultados, com os gráficos e outras análises que fizermos)

## References
(#TODO Sessão de referências com todos os links e possíveis artigos que podemos ligar ao nosso projeto)