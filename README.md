# Predicting Future NBA Game Winners

## Overview 
This project predicts the winner of the next matchup between two NBA teams using game data scraped from https://www.basketball-reference.com/ for NBA games from the 2009 - 2010 season to the 2023 - 2024 season (2024-2025 season will be added soon). The pipeline collects raw HTML, parses team/game statistics, engineers features (elo ratings, rolling averages, rest advantage, head to head stats, etc), and trains machine learning models to output a pre game win probability. 

# Current Results 
- Logistic Regression (baseline): Validation Accuracy = 55.99% (AUC = 0.575)
- Logistic Regression (v2, including engineered features): Validation Accuracy = 62.84% (AUC = 0.675)
- Deep Neural Network: Validation Accuracy = 62.46% (AUC = 0.6739)
- LSTM RNN (initial model): Validation Accuracy = 52.72%

## Technologies Used 
- **Python**
- **Playwright**
- **BeautifulSoup**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **PyTorch**
- **XGBoost**

## XGBoost 
For this project I used XGBoost for classification due to its high accuracy and performance and ability to handle large datasets. XGBoost is a type of gradient boosting algorithm that builds decision trees sequentially with each tree focusing on the mistakes made by the previous trees. What makes XGBoost an extreme gradient boosting alogrithm is its ability to optimize tradional gradient boosting which makes it faster and more accurate. It uses parallel processing by splitting the task of evaluating the best splits for the features across multiple CPU cores. They work at the same time to find the best splits for their assigned features which reduces the time to build the decision trees making XGBoost faster than traditional gradient boosting. XGBoost also uses L1 and L2 regularization to prevent overfitting. They add pentalty terms to the loss function which prevents the model from assigning too much importance to one feature. L1 regularization adds the absolute values of the weights of the features to the loss function. This encourages the model to set some of the weights of the features to 0 as it wants to minimize the loss function. This simplifies the model and eliminates less important features. L2 regularization adds the squares of the weights of the features to the loss function which encourages the model to evenly distribute the importance of the features and prevents the weights from growing too large. These techniques ensure the model generalizes well to new data and makes XGBoost a powerful model. 

## Project Structure 

### 'scrape_nba_data.py'
Handles the web scraping of NBA monthly game schedules and box scores from Basketball-Reference. 

#### Key Components:
- **Playwright**: Playwright opens the Hoops Reference website automating the process of opening the monthly schedule pages from the 2009 - 2010 season to the 2023 - 2024 season. Playwright gathers all the urls and then navigates to each box score page for scraping. 
- **BeautifulSoup**: BeatifulSoup parses the HTML content to extract the games schedules and the box scores scraping data such as the teams, scores, team statistics, player statistics, etc.
- **Data Storage**: After the data is scraped the monthly schedule HTML files are stored in the `nba_data/monthly_schedule` directory and the box scores HTML files are stored in the `nba_data/scores` directory.

### 'parse_data.py'
Processes the scraped HTML data and transforms it into a Pandas dataframe. 

#### Key Functions: 
- **'parse_html'**: This function  opens and parses each HTML file with BeautifulSoup. It removes the table rows with classes 'over_header' and 'thead' and returns the cleaned BeautifulSoup object ready for further data extraction.
- **'read_scoring_summary**: Extracts the scoring summary tables for each game (points per quarter and total) and stores it in a dataframe. 
- **'read_box_scores**: Extracts basic box scores table and the advanced box score tables for the given teams and formats it into a dataframe. This data includes basic and advanced team and player statistics. 

The basic and advanced stats for reach team are stored in seperate dataframes. Then the last row of both the basic and advanced dataframes which contain the total team stats are concatenated together to form the totals dataframe. Then the script concatenates the maximum values from both the basic and advanced dataframes. Then the totals and maxes dataframes are concatenated into a summary dataframe. The summary data frame and scoring summary data frame are then concatenated creating the game dataframe. Labels are assigned for the home and away teams and a game_opp dataframe is created in which opponent stats are mirrored. The game dataframe is then concatenated with the game_opp dataframe resulting in a single dataframe that contains both the team and opponent stats in one row. Season and date information is added and the resulting dataframe is saved as 'nba_game_data.csv'. 

### 'cleanup_nba_data.py'
Processes the 'nba_game_data.csv' and cleans the data. A 'game_id' column for each game is created by combining the game date and team names. A binary win column is added and empty and unneccesary columns are removed and the cleaned data is finally saved as 'nba_game_data_updated.csv'.

### 'restructure_and_preprocess.ipynb' 
Preprocessing stage where data is reshaped for modeling. Currently each row represents a matchup between two teams and the stats from that game. 
- `Team_A` and `Team_B` columns are created (alphabetical ordering across matchups). Each team is then encoded with numbers from 0-29 each number representing a different team. 
- A new `matchup_id` is created replacing the old one removing the date and keeping just team names (Team_A vs Team_B)
- `Team_A_home` and `playoff_game` indicators created for the game. 
- `Team_A_win_next` label created indicating whether Team_A wins the next matchup with Team_B.
- Finally sequences are created for the LSTM RNN. For each game the last 10 games between those teams are created into sequences and stored in `x_sequence` and `y_sequence` which are then converted to numpy arrays (samples, timesteps, fearures) for LSTM input. We end up with two arrays `x_sequence` and `y_sequence`. 
- New dataframe (before sequences created) also saved as `nba_game_df.csv`.

### `feature_engineering.ipynb` 
Engineered new and stronger predictive features. 
- `next_matchup_date` column added for each game (row).
- Create rolling averages of last 5 and last 10 games both teams have played before their next matchup date.
- Elo for each team is calculated for after the last game they have played and before their next matchup date using the FiveThirtyEight formula including season reversions.
- Season win percentages are added for each team after each game (the game in row the last head to head match they played) and the season win percentages going into their next matchup.
- Head to head win percentages are added.
- Back to back indicators for next matchup added for both teams.
- Number of days of rest before next matchup added for both teams and rest advantage between each team is calculated.
- Rows (games) where the next matchup information is not in the dataframe is dropped (where `Team_A_win_next` and `next_matchup_date` are both null).
- Once all new features are created pearson correlation is calculated between all numerical features and redudant features are carefully dropped to reduce multicollinearity.
- Distance correlation is calculated between every numerical feature and the label `Team_A_win_next_matchup` to measure the linear and nonlinear relationships between each feature and the label. Along with this, the chi square statistic is calculated between the categorical features and the label  `Team_A_win_next_matchup`. Finally features with low signals are dropped and the final data frame is saved as `nba_engineered_game_df.csv`. 
  
### Model Training and Evaluation 
In this script the XGBoost Classification  model is built to predict the winner of NBA games. The features used include the basic team stats, advanced team stats, and the player max sats. To view a full list of each exact feature refer to the 'xgboost_model.py'.

#### Model Parameters 
- **`objective='binary:logistic'`:**
  - Logistic regression is used for binary classification (predicting win or loss)
-  **`n_estimators=1000`:**
   - The model will build up to 1000 trees. 
-  **`learning_rate=0.05`:**
   - The learning rate controls the impact each new tree has on the model. A small learning rate like 0.05 reduces the impact each tree has which lets the model learn more slowly which prevents overfitting leading to a more accurate model. 
- **`eval_metric="logloss"`:**
  -  Logistic Loss also known as cross entropy loss is used to measure the error between the predicted probability that the team will win or lose and the actual label.
- **`early_stopping_rounds=20`:**
  - Early stopping rounds is used to prevent overfitting. The model will stop traiing if the log loss does not decrease for 20 consecutive rounds. 

## Performance 
The model was trained using a 80-20 train and test split. The model achieved a test accuracy of **99.88%** and reduced the log loss from 0.65505 to 0.00325
