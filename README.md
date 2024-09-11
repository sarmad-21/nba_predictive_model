# NBA Predictive XGBoost Model

## Overview 
This project involves developing a XGBoost model to predict the outcome of NBA games based on various statistics. The data is scraped from https://www.basketball-reference.com/ for NBA seasons from the 2009 - 2010 season to the 2023 - 2024 season. The data is cleaned and organized in to a Pandas dataframe before being trained with an XGBoost Classifer model. The model achieved a test accuracy of 99.88% making it highly accurate in predicting the winner of NBA games. 

## Technologies Used 
- **Python**
- **Playwright**
- **BeautifulSoup**
- **Pandas**
- **Scikit-learn**
- **XGBoost**

## XGBoost 
For this project I used XGBoost for classification due to its high accuracy and performance and ability to handle large datasets. XGBoost is a type of gradient boosting algorithm that builds decision trees sequentially with each tree focusing on the mistakes made by the previous trees. What makes XGBoost an extreme gradient boosting alogrithm is its ability to optimize tradional gradient boosting which makes it faster and more accurate. It uses parallel processing by splitting the task of evaluating the best splits for the features across multiple CPU cores. They work at the same time to find the best splits for their assigned features which reduces the time to build the decision trees making XGBoost faster than traditional gradient boosting. XGBoost also uses L1 and L2 regularization to prevent overfitting. They add pentalty terms to the loss function which prevents the model from assigning too much importance to one feature. L1 regularization adds the absolute values of the weights of the features to the loss function. This encourages the model to set some of the weights of the features to 0 as it wants to minimize the loss function. This simplifies the model and eliminates less important features. L2 regularization adds the squares of the weights of the features to the loss function which encourages the model to evenly distribute the importance of the features and prevents the weights from growing too large. These techniques ensure the model generalizes well to new data and makes XGBoost a powerful model. 

## Project Structure 

### 'scrape_nba_data.py'
This script handles the web scraping of NBA monthly game schedules and box scores from Basketball-Reference. 
