import os
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import numpy as np

SCORE_DIR = "nba_data/scores"
box_scores = os.listdir(SCORE_DIR)
print(len(box_scores))
box_scores = [os.path.join(SCORE_DIR, f) for f in box_scores if f.endswith(".html")]

def parse_html (box_score):
    with open(box_score) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    [s.extract() for s in soup.select("tr.over_header, tr.thead")]
    return soup

def read_scoring_summary(soup):
    html_string = str(soup)
    html_io = StringIO(html_string)  # Create a StringIO object to mimic a file
    table = pd.read_html(html_io, attrs={"id": "line_score"})[0]
    table.columns = ['Team'] + [f'Q{i}' for i in range(1, len(table.columns) - 1)] + ['Total']
    return table

def read_box_scores(soup, team, stat):
    html_string= str(soup)
    html_io = StringIO(html_string)
    df = pd.read_html(html_io, attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[0]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how='all',axis=0)
    df.columns = [f'{col}_{stat}' for col in df.columns]
    return df

def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a.get("href") for a in nav.find_all('a') if a.get("href")]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season

games = []

for bs in box_scores:
    soup = parse_html(bs)
    scoring_summary = read_scoring_summary(soup)
    teams = list(scoring_summary["Team"])
        #if scoring_summary is not None:
            #print(teams)
            #print(scoring_summary)
    summaries = []
    for team in teams:
        basic = read_box_scores(soup, team, "basic")
        advanced = read_box_scores(soup, team, "advanced")
        # Filter out unwanted columns from both basic and advanced DataFrames
        basic = basic.loc[:, ~basic.columns.str.contains('unnamed', case=False)]
        advanced = advanced.loc[:, ~advanced.columns.str.contains('unnamed', case=False)]
        totals = pd.concat([basic.iloc[-1, :], advanced.iloc[-1, :]])
        totals.index = [f"{x}_totals" for x in totals.index]
        maxes = pd.concat([basic.iloc[:-1, :].max(), advanced.iloc[:-1, :].max()]) #highest number individual player had in game
        maxes.index = [f"{x}_max" for x in maxes.index]
        summary = pd.concat([totals, maxes])
        summaries.append(summary)

    if summaries:
        summary_df = pd.concat(summaries, axis=1).T.reset_index(drop=True)
        game = pd.concat([summary_df, scoring_summary.reset_index(drop=True)], axis=1)
        num_teams = len(teams)
        game["home"] = np.tile([0, 1], num_teams // 2)[:num_teams]  # Alternates 0, 1 for each team if more than two teams
        game_opp = game.iloc[::-1].reset_index(drop=True)
        game_opp.columns = [str(col) + '_opp' for col in game_opp.columns]
        full_game = pd.concat([game, game_opp], axis=1)
        full_game["season"] = read_season_info(soup)
        full_game["date"] = os.path.basename(bs)[:8]
        full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")
        games.append(full_game)

    if len(games) % 100 ==0 and len(games) > 0:
        print(f"{len(games)} / {len(box_scores)}")

try:
    games_df = pd.concat(games, ignore_index=True)
    games_df.to_csv('nba_game_data.csv', index=False)
    print(games_df)
except Exception as e:
    print("Failed to concatenate games:", str(e))


1
