import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def main():
    # Load the data
    data = pd.read_csv("./data/Pitching.csv")

    # Drop rows with missing values in the selected features
    features = ['W', 'L', 'G', 'GS', 'CG', 'SHO', 'SV', 'IPouts', 'H', 'ER', 'BB', 'SO', 'ERA', 'R']
    data.dropna(subset=features, inplace=True)

    # Select features and target variable
    features_data = data[features]
    target = data['HR']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_data, target, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the KNN model
    k = 5
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = knn_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Make predictions on new data
    new_data = pd.DataFrame([[12, 15, 30, 30, 30, 0, 0, 639, 500, 103, 31, 15, 4.5, 292]])
    new_data_scaled = scaler.transform(new_data)
    prediction = knn_model.predict(new_data_scaled)
    print("Predicted HR:", prediction)


if __name__ == "__main__":
    main()
