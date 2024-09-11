import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier


data = pd.read_csv('nba_game_data_updated.csv')
data = data.drop(columns=['MP_advanced_totals'])
data = data.drop(columns=['MP_advanced_totals_opp'])
data.rename(columns={'MP_basic_totals_opp': 'MP'}, inplace=True)
print(data.shape)
print(data.columns.tolist())

basic_features = [
    'FG_basic_totals', 'FGA_basic_totals', 'FG%_basic_totals', '3P_basic_totals', '3PA_basic_totals',
    '3P%_basic_totals', 'FT_basic_totals', 'FTA_basic_totals', 'FT%_basic_totals', 'ORB_basic_totals',
    'DRB_basic_totals', 'TRB_basic_totals', 'AST_basic_totals', 'STL_basic_totals', 'BLK_basic_totals',
    'TOV_basic_totals', 'PF_basic_totals', 'PTS_basic_totals', 'FG_basic_totals_opp', 'FGA_basic_totals_opp',
    'FG%_basic_totals_opp', '3P_basic_totals_opp', '3PA_basic_totals_opp', '3P%_basic_totals_opp',
    'FT_basic_totals_opp', 'FTA_basic_totals_opp', 'FT%_basic_totals_opp', 'ORB_basic_totals_opp',
    'DRB_basic_totals_opp', 'TRB_basic_totals_opp', 'AST_basic_totals_opp', 'STL_basic_totals_opp', 'BLK_basic_totals_opp',
    'TOV_basic_totals_opp', 'PF_basic_totals_opp', 'PTS_basic_totals_opp', 'TS%_advanced_totals', 'eFG%_advanced_totals',
    '3PAr_advanced_totals', 'FTr_advanced_totals', 'ORB%_advanced_totals', 'DRB%_advanced_totals', 'TRB%_advanced_totals',
    'AST%_advanced_totals', 'STL%_advanced_totals', 'BLK%_advanced_totals', 'TOV%_advanced_totals', 'USG%_advanced_totals',
    'ORtg_advanced_totals', 'DRtg_advanced_totals', 'TS%_advanced_totals_opp', 'eFG%_advanced_totals_opp', '3PAr_advanced_totals_opp',
    'FTr_advanced_totals_opp', 'ORB%_advanced_totals_opp', 'DRB%_advanced_totals_opp', 'TRB%_advanced_totals_opp',
    'AST%_advanced_totals_opp', 'STL%_advanced_totals_opp', 'BLK%_advanced_totals_opp', 'TOV%_advanced_totals_opp',
    'USG%_advanced_totals_opp', 'ORtg_advanced_totals_opp', 'DRtg_advanced_totals_opp', 'Q1', 'Q2', 'Q3', 'Q4', 'OT5', 'OT6', 'OT7', 'OT8',
    'Total', 'home', 'season', 'MP', 'Q1_opp', 'Q2_opp', 'Q3_opp', 'Q4_opp', 'OT5_opp', 'OT6_opp', 'OT7_opp', 'OT8_opp', 'Total_opp',
]

player_max_features = [
    'FG_basic_max', 'FGA_basic_max', 'FG%_basic_max', '3P_basic_max', '3PA_basic_max', '3P%_basic_max', 'FT_basic_max', 'FTA_basic_max',
    'FT%_basic_max', 'ORB_basic_max', 'DRB_basic_max', 'TRB_basic_max', 'AST_basic_max', 'STL_basic_max', 'BLK_basic_max', 'TOV_basic_max',
    'PF_basic_max', 'PTS_basic_max', 'GmSc_basic_max', '+/-_basic_max', 'TS%_advanced_max', 'eFG%_advanced_max', '3PAr_advanced_max',
    'FTr_advanced_max', 'ORB%_advanced_max', 'DRB%_advanced_max', 'TRB%_advanced_max', 'AST%_advanced_max', 'STL%_advanced_max',
    'BLK%_advanced_max', 'TOV%_advanced_max', 'USG%_advanced_max', 'ORtg_advanced_max', 'DRtg_advanced_max', 'FG_basic_max_opp',
    'FGA_basic_max_opp', 'FG%_basic_max_opp', '3P_basic_max_opp', '3PA_basic_max_opp', '3P%_basic_max_opp', 'FT_basic_max_opp',
    'FTA_basic_max_opp', 'FT%_basic_max_opp', 'ORB_basic_max_opp', 'DRB_basic_max_opp', 'TRB_basic_max_opp', 'AST_basic_max_opp',
    'STL_basic_max_opp', 'BLK_basic_max_opp', 'TOV_basic_max_opp', 'PF_basic_max_opp', 'PTS_basic_max_opp', 'GmSc_basic_max_opp',
    '+/-_basic_max_opp', 'TS%_advanced_max_opp', 'eFG%_advanced_max_opp', '3PAr_advanced_max_opp', 'FTr_advanced_max_opp',
    'ORB%_advanced_max_opp', 'DRB%_advanced_max_opp', 'TRB%_advanced_max_opp', 'AST%_advanced_max_opp', 'STL%_advanced_max_opp',
    'BLK%_advanced_max_opp', 'TOV%_advanced_max_opp', 'USG%_advanced_max_opp', 'ORtg_advanced_max_opp', 'DRtg_advanced_max_opp'
]

x = data[basic_features + player_max_features]
y = data['win']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = XGBClassifier(objective='binary:logistic', n_estimators=1000, learning_rate=0.05,
                      eval_metric="logloss", early_stopping_rounds=20, verbose=True)
model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"The model achieves a test accuracy of {accuracy*100} %")


