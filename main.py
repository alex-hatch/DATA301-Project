import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from pyfiglet import Figlet
from tqdm import tqdm
import time


def loading_animation():
    for _ in tqdm(range(4), desc="Predicting home runs", position=0, leave=True):
        time.sleep(0.5)


def knn_prediction(data):
    # Drop rows with missing values in the selected features
    features = ['W', 'L', 'H']
    data.dropna(subset=features, inplace=True)

    # Select features and target variable
    features_data = data[features]
    target = data['HR']

    # Normalize the features data
    scaler = MinMaxScaler()
    features_data_normalized = scaler.fit_transform(features_data)

    # Split the normalized data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_data_normalized, target, test_size=0.2,
                                                        random_state=42)

    # Train the KNN model
    k = 5
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Get new data from user
    user_wins = int(input("How many wins does this pitcher have?\n>>> "))
    user_losses = int(input("How many losses does this pitcher have?\n>>> "))
    user_hits = int(input("How many hits has this pitcher given up?\n>>> "))

    # Normalize the new data
    new_data = pd.DataFrame([[user_wins, user_losses, user_hits]])
    new_data_scaled = scaler.transform(new_data)

    # Make prediction using the normalized new data
    prediction = knn_model.predict(new_data_scaled)
    loading_animation()
    print("Predicted HR:", prediction[0])
    print("Mean Squared Error:", mse)


def load_data(file_path):
    return pd.read_csv(file_path)


def print_ascii():
    figlet = Figlet(font="starwars")

    pitching = figlet.renderText("PITCHING")
    statistics = figlet.renderText("STATS")

    print(pitching)
    print(statistics)


def main():
    data = load_data("./data/Pitching.csv")
    print_ascii()

    while True:
        print("Select an option from the menu below")
        print("Type 'quit' to terminate program")
        print("\n-------------------------------\n")
        print("1. Print ascii art")
        print("2. Predict how many home runs a pitcher gives using KNN")

        user_input = input(">>> ")

        if user_input == '1':
            print_ascii()
        elif user_input == '2':
            knn_prediction(data)
        elif user_input == '':
            continue
        elif user_input == 'quit':
            break


if __name__ == "__main__":
    main()
