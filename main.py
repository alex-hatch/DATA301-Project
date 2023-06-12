import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pyfiglet import Figlet
import random
import math
from numpy.random import permutation
from tqdm import tqdm
import time
import numpy as np

def linear_regression(data, year, height):
    # Get only the data frame that year
    data = data[data['season'] == year]

    # Feature and target
    cols = ['player_height', 'pts']

    # Filter the data frame
    data = data[cols]

    # Normalize the data
    # data = normalize(data, cols)

    # Extract features and target variable
    X = data[cols[:-1]]
    y = data[cols[-1]]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    """
    # Evaluate the model
    r2_score = model.score(X_test, y_test)
    mse = np.mean((y_pred - y_test) ** 2)

    print(f'R^2 Score: {r2_score:.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
    """

    new_data = pd.DataFrame([[height]])

    prediction = model.predict(new_data)

    return prediction[0]

def knn_prediction(data, year, height):
    data = data[data['season'] == year]
    cols = ['player_height', 'pts']

    # Filter the data frame
    data = data[cols]
    X = data[cols[:-1]]
    y = data[cols[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the KNN model
    k = 5
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = knn_model.predict(X_test)
    mse = (abs(y_pred - y_test)).mean()
    print("Mean Squared Error:", mse)

    # Normalize the new data
    new_data = pd.DataFrame([[height]])
    prediction = knn_model.predict(new_data)

    # Make prediction using the normalized new data
    print("Predicted points from KNN model:", prediction[0])

def height_distribution(data):
    return (data['player_height'].quantile(0.25),
            data['player_height'].quantile(0.5),
            data['player_height'].quantile(0.75))

def fix_season(attribute):
    return attribute.split('-')[0]

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['season'] = df['season'].apply(fix_season)
    df = df[df['gp'] > 10]
    return df

def print_ascii():
    figlet = Figlet(font="starwars")

    pitching = figlet.renderText("BBALL")
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
        print("2. Predict how many points a players gets using KNN")
        print("3. Predict how many points a players gets using Linear Regression")
        print("4. Height Distribution")

        user_input = input(">>> ")

        if user_input == '1':
            print_ascii()
        elif user_input == '2':
            result = {'Year': [],
                      'Points': []}
            years = np.arange(1996, 2022)
            height = int(input("What is their height (in cm)?\n>>> "))
            for year in years:
                res = knn_prediction(data, str(year), height)
                result['Year'].append(year)
                result['Points'].append(res)
            plt.scatter(result['Year'], result['Points'])
            plt.xlabel('Year')
            plt.ylabel('Average Points Per Game')
            plt.title('Average points per game as a function of the year at ' + str(height) + 'cm')
            plt.show()
        elif user_input == '3':
            result = {'Year': [],
                      'Points': []}
            years = np.arange(1996, 2022)
            height = int(input("What is their height (in cm)?\n>>> "))
            for year in years:
                res = linear_regression(data, str(year), height)
                result['Year'].append(year)
                result['Points'].append(res)
            plt.scatter(result['Year'], result['Points'])
            plt.xlabel('Year')
            plt.ylabel('Average Points Per Game')
            plt.title('Average points per game as a function of the year at ' + str(height) + 'cm')
            plt.show()
        elif user_input == '':
            continue
        elif user_input == 'quit':
            break

        print("\n\n\n\n")


if __name__ == "__main__":
    main()
