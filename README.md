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
This script handles the web scraping of NBA monthly game schedules and box scores from Basketball-Reference. In this script I have used the liibraries PlayWright and BeatuifulSoup for web scraping. 

#### Key Components:
- **Playwright**: Playwright opens the Hoops Reference website automating the process of opening the monthly schedule pages from the 2009 - 2010 season to the 2023 - 2024 season. Playwright then gathers all the urls and then navigates to the box score page for each game before proccesing with data scraping.
- **BeautifulSoup**: BeatifulSoup parses the HTML content to extract the games schedules and the box scores scraping data such as the teams, scores, team statistics, player statistics, etc.
- **Data Storage**: After the data is scraped the monthly schedule HTML files are stored in the `nba_data/monthly_schedule` directory and the box scores HTML files are stored in the `nba_data/scores` directory.

### 'parse_data.py'
The script processes the scraped HTML data and transforms it into a Pandas dataframe that is used to build our model. 

#### Key Functions: 
- **'parse_html'**: This function  opens and parses each HTML file with BeautifulSoup. It removes the table rows with classes 'over_header' and 'thead' and returns the cleaned BeautifulSoup object ready for further data extraction.
- **'read_scoring_summary**: Extracts the scoring summary tables for each game which include the points per quarter for each team and the total points each team scored and stores it in a dataframe. 
- **'read_box_scores**: Extracts data from the basic box scores table and the advanced box score tables for the given teams and formats it into a dataframe. This data includes basic and advanced team and player statistics.

The basic and advanced stats for reach team are stored in seperate dataframes. Then the last row of both the basic and advanced dataframes which contain the total team stats are concatenated together to form the totals dataframe. Then the script concatenates the maximum values from both the basic and advanced dataframes. Then the totals and maxes dataframes are concatenated into a summary dataframe. The summary data frame and scoring summary data frame are then concatenated creating the game dataframe. Labels are assigned for the home and away teams and a game_opp dataframe is created in which opponent stats are mirrored. The game dataframe is then concatenated with the game_opp dataframe resulting in a single dataframe that contains both the team and opponent stats in one row. Season and date information is added and the resulting dataframe is saved as 'nba_game_data.csv'. 

### 'cleanup_nba_data.py'
This script processes the 'nba_game_data.csv' and cleans the data. A game id column for each game is created by combining the game date and team names. A win column is added which assigns lables based on who won and lost the game. Then empty and unneccesary columns are removed and the cleaned data is finally saved as 'nba_game_data_updated.csv'.

### XGBoost Classification Model ('xgboost_model.py')
In this script the XGBoost Classification  model is built to predict the winner of NBA games. The features used include the basic team stats, advanced team stats, and the player max sats. To view a full list of each exact feature refer to the 'xgboost_model.py'.

#### Model Parameters 
- **`objective='binary:logistic'`:**
  - Logistic regression is used for binary classification (predicting win or loss)
-  **`n_estimators=1000`:**
  - The model will build up to 1000 trees
-  **`learning_rate=0.05`:**
  - The learning rate controls the impact each new tree has on the model. A small learning rate like 0.05 reduces the impact each tree has which lets the model learn more slowly which prevents overfitting leading to a more accurate model. 
- **`eval_metric="logloss"`:**
  -  Logistic Loss also known as cross entropy loss is used to measure the error between the predicted probability that the team will win or lose and the actual label.
- **`early_stopping_rounds=20`:**
  - Early stopping rounds is used to prevent overfitting. The model will stop traiing if the log loss does not decrease for 20 consecutive rounds. 

## Performance 
The model was trained using a 80-20 train and test split. The model achieved a test accuracy of **99.88%** and reduced the log loss from 0.65505 to 0.00325
