import pandas as pd


data = pd.read_csv("nba_game_data.csv", index_col=0)
print(data)
print(data.shape)
data['game_id'] = data.apply(lambda row: f"{row['date']}_{'_'.join(sorted([row['Team'], row['Team_opp']]))}", axis=1)
print(data[['game_id', 'Team', 'Total', 'Total_opp']].head())

def add_outcome_column(df):
    df['win'] = 'Unknown'
    grouped = df.groupby('game_id')
    for game_id, group in grouped:
        if len(group) == 2:
            team1 = group.iloc[0]
            team2 = group.iloc[1]
            team1_total = team1['Total']
            team2_total = team2['Total']
            if team1_total > team2_total:
                df.loc[(df['game_id'] == game_id) & (df['Team'] == team1['Team']), 'win'] = '1'
                df.loc[(df['game_id'] == game_id) & (df['Team'] == team2['Team']), 'win'] = '0'
            elif team1_total < team2_total:
                df.loc[(df['game_id'] == game_id) & (df['Team'] == team1['Team']), 'win'] = '0'
                df.loc[(df['game_id'] == game_id) & (df['Team'] == team2['Team']), 'win'] = '1'

add_outcome_column(data)
data = data.sort_values(by=['date', 'game_id'])
data = data.drop(columns=['BPM_advanced_max_opp', 'BPM_advanced_max', 'MP_advanced_max', 'MP_advanced_max_opp', 'MP_basic_max_opp', 'BPM_advanced_totals_opp', 'MP_basic_max', 'BPM_advanced_totals', 'GmSc_basic_totals', '+/-_basic_totals', 'GmSc_basic_totals_opp', '+/-_basic_totals_opp'])
print(data[['game_id', 'Team', 'Total', 'Total_opp', 'win']])
columns_to_fill = ['Q5', 'Q6', 'Q7', 'Q8', 'Q5_opp', 'Q6_opp', 'Q7_opp', 'Q8_opp']
data[columns_to_fill] = data[columns_to_fill].fillna(0)
rename_dict = {
    'Q5': 'OT5',
    'Q6': 'OT6',
    'Q7': 'OT7',
    'Q8': 'OT8',
    'Q5_opp': 'OT5_opp',
    'Q6_opp': 'OT6_opp',
    'Q7_opp': 'OT7_opp',
    'Q8_opp': 'OT8_opp',
}

data = data.rename(columns=rename_dict)
data = data.reset_index(drop=True)
data.to_csv("nba_game_data_updated.csv")
null_indices = data[data.isnull().any(axis=1)]
for index, row in null_indices.iterrows():
    null_columns = row[row.isnull()].index.tolist()
    print(f"Index {index} has null values in columns: {', '.join(null_columns)}")

data.loc[35346, '+/-_basic_max'] = 5.0
data.loc[35346, '+/-_basic_max_opp'] = 28.0
data.loc[35347, '+/-_basic_max'] = 28.0
data.loc[35347, '+/-_basic_max_opp'] = 5.0

data.loc[35348, '+/-_basic_max'] = 6.0
data.loc[35349, '+/-_basic_max_opp'] = 6.0

data.loc[35352, '+/-_basic_max'] = 6.0
data.loc[35352, '+/-_basic_max_opp'] = 17.0
data.loc[35353, '+/-_basic_max'] = 17.0
data.loc[35353, '+/-_basic_max_opp'] = 6.0

data.loc[35354, '+/-_basic_max'] = 12.0
data.loc[35354, '+/-_basic_max_opp'] = 27.0
data.loc[35355, '+/-_basic_max'] = 27.0
data.loc[35355, '+/-_basic_max_opp'] = 12.0

data.loc[35356, '+/-_basic_max'] = 20.0
data.loc[35356, '+/-_basic_max_opp'] = 5.0
data.loc[35357, '+/-_basic_max'] = 5.0
data.loc[35357, '+/-_basic_max_opp'] = 20.0

data.loc[35358, '+/-_basic_max'] = 15.0
data.loc[35358, '+/-_basic_max_opp'] = 3.0
data.loc[35359, '+/-_basic_max'] = 3.0
data.loc[35359, '+/-_basic_max_opp'] = 15.0

data.loc[35360, '+/-_basic_max'] = 8.0
data.loc[35360, '+/-_basic_max_opp'] = 12.0
data.loc[35361, '+/-_basic_max'] = 12.0
data.loc[35361, '+/-_basic_max_opp'] = 8.0

data.loc[35362, '+/-_basic_max'] = 22.0
data.loc[35362, '+/-_basic_max_opp'] = 1.0
data.loc[35363, '+/-_basic_max'] = 1.0
data.loc[35363, '+/-_basic_max_opp'] = 22.0

data.loc[36326, '+/-_basic_max'] = 5.0
data.loc[36326, '+/-_basic_max_opp'] = 24.0
data.loc[36327, '+/-_basic_max'] = 24.0
data.loc[36327, '+/-_basic_max_opp'] = 5.0

data.loc[38034, 'FT%_basic_totals'] = 0.0
data.loc[38034, 'FT%_basic_max'] = 0.0
data.loc[38035, 'FT%_basic_totals_opp'] = 0.0
data.loc[38035, 'FT%_basic_max_opp'] = 0.0

data.to_csv("nba_game_data_updated.csv", index=True)
print("\nAfter update:")
null_indices = data[data.isnull().any(axis=1)]
for index, row in null_indices.iterrows():
    null_columns = row[row.isnull()].index.tolist()
    print(f"Index {index} has null values in columns: {', '.join(null_columns)}")
