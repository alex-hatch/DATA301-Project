import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pyfiglet import Figlet
from tqdm import tqdm
import time
import numpy as np

def top30height(data):
    sorted_vals = data.sort_values(by=['season', 'pts'], ascending=False)
    groupings = sorted_vals.groupby('season').head(30)
    meanval = groupings.groupby('season', as_index=False)['player_height'].mean().iloc[1:]
    # print(sorted_vals[['player_name', 'pts', 'season']])
    meanval['years'] = meanval['season'].str[0:4]
    meanval2 = meanval.iloc[:, [1, 2]]
    # Meanval2 = meanval.loc['player_height', 'years']
    print(meanval2)
    fig, ax = plt.subplots()
     # Generate a bar plot on the axes
    ax.bar(meanval2.iloc[:, 1], meanval2.iloc[:, 0])

    # (Optional) Add labels and title
    ax.set_xlabel('season')
    ax.set_ylabel('player_height')
    ax.set_title('Top 30 Players Avg Height From Every Season')

    plt.ylim(197, 204)

    # Show the plot
    plt.show()

def heightVsRebScatter(data): 
    # - Take a single old season and plot their height and weight. Do the same for a new season and compare the clusters.
    old_season = data[data['season'] == '1998-99']
    new_season = data[data['season'] == '2019-20']
    # sorted_old = data.sort_values(by=['season', 'pts'], ascending=False)
    # sorted_new = data.sort_values(by=['season', 'pts'], ascending=False)
    old = old_season.groupby('season').head(100)
    new = new_season.groupby('season').head(100)
    old.plot.scatter(x='reb', y='player_height', s=2)
    new.plot.scatter(x='reb', y='player_height', s=2)

    plt.ylim(180, 230)
    plt.xlim(0, 16)
    plt.show()

def scatterplot2(data):
    # data['Greater'] = data[data['player_height'] > 198]
    sorted_vals = data.sort_values(by=['season', 'pts'], ascending=False)
    groupings = sorted_vals.groupby('season').head(30)
    groupingGreater = groupings[groupings['age'] < 24]
    groupingGreater['Count'] = 0
    greaterCounts = groupingGreater.groupby('season').count().reset_index()
    greaterCounts['years'] = greaterCounts['season'].str[0:4]
    counts = greaterCounts.loc[:,['Count', 'years']]
    print(counts)


    fig, ax = plt.subplots()
     # Generate a bar plot on the axes
    ax.bar(counts.iloc[:, 1], counts.iloc[:, 0])

    # (Optional) Add labels and title
    ax.set_xlabel('Season')
    ax.set_ylabel('Number of Players')
    ax.set_title('Number of top 30 players over 6ft 5in from each Season')
    # plt.ylim(15, 25)

    # Show the plot
    plt.show()


def shortPlayersAvgRebs(data):
    #Take all players under 6’5. Group by season. Average number of rebounds for each player
    
    groupingGreater = data[data['player_height'] < 210]
    g = groupingGreater[groupingGreater['net_rating'] > 0]
    g1 = g[g['player_weight'] < 100]
    groupings = g1.groupby('season')['reb'].mean().reset_index()
    groupings['years'] = groupings['season'].str[0:4]
    fig, ax = plt.subplots()

     # Generate a bar plot on the axes
    ax.bar(groupings.loc[:,'years'], groupings.loc[:, 'reb'])

    # (Optional) Add labels and title
    ax.set_xlabel('Season')
    ax.set_ylabel('Average Rebounds')
    ax.set_title('Average Rebounds for players under 6ft 5in cm from each Season')
    # plt.ylim(1, 3)

    # Show the plot
    plt.show()

def lineplotNetRating(data):
    groupings = data[data['player_height'] < 198]
    groupings2 = groupings[groupings['player_weight'] < 105]
    groupings3 = groupings2[groupings2['net_rating'] > 0]
    g = groupings3.groupby('season')['net_rating'].mean().reset_index()
    g['years'] = g['season'].str[0:4]
    g.plot(x = 'years', y = 'net_rating')
    # g.plot(g.loc[:,'years'], g.loc[:, 'net_rating'])
    
    # plt.ylim(70, 140)
    # plt.xlim(160, 230)
    plt.show()

def lineplotAge(data):
    sorted_vals = data.sort_values(by=['season', 'pts'], ascending=False)
    g = sorted_vals.groupby('season')['player_weight'].mean().reset_index()
    g['years'] = g['season'].str[0:4]
    g.plot(x = 'years', y = 'player_weight')
    # g.plot(g.loc[:,'years'], g.loc[:, 'net_rating'])
    
    # plt.ylim(70, 140)
    # plt.xlim(160, 230)
    plt.show()
    

def load_data(file_path):
    return pd.read_csv(file_path)


def print_ascii():
    figlet = Figlet(font="starwars")

    pitching = figlet.renderText("PITCHING")
    statistics = figlet.renderText("STATS")

    print(pitching)
    print(statistics)


def main():
    data = load_data("./data/all_seasons.csv")
    print_ascii()

    while True:
        print("Select an option from the menu below")
        print("Type 'quit' to terminate program")
        print("\n-------------------------------\n")
        print("1. Print ascii art")
        print("2. Predict how many home runs a pitcher gives using KNN")
        print("3. Predict how many home runs a pitcher gives using Linear Regression")
        print("4. Top 30 Players Avg Height Over Seasons")
        print('5. Height vs Rebounds Scatter for 1997 vs 2021')
        print('6. Players Under 6 ft 5in Average Rebounds')
        print('7. Net Rating For Short And Less Heavier Players')
        print('8. LinePlot of Player Weight Over the Seasons')

        user_input = input(">>> ")

        if user_input == '1':
            print_ascii()
        elif user_input == '2':
            knn_prediction(data)
        elif user_input == '3':
            linear_regression(data)
        elif user_input == '4':
            top30height(data)
        elif user_input == '5':
            heightVsRebScatter(data)
        # elif user_input == '7':
        #     scatterplot2(data)
        elif user_input == '6':
            shortPlayersAvgRebs(data)
        elif user_input == '7':
            lineplotNetRating(data)
        elif user_input == '8':
            lineplotAge(data)
        elif user_input == '':
            continue
        elif user_input == 'quit':
            break

        print("\n\n\n\n")


if __name__ == "__main__":
    main()
