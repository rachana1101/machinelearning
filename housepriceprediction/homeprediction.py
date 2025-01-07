#
# Reference: From Kaggle - Machine learning simple project
#

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
def main():

    iowa_file_path = 'melb_data.csv'

    home_data = pd.read_csv(iowa_file_path)

    print(home_data.columns)

    y = home_data.Price

    feature_names = ['Rooms', 'Bedroom2', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt']

    # Select data corresponding to features in feature_names
    X = home_data[feature_names]

    # Review data
    # print description or statistics from X
    print(X.describe())

    # print the top few lines
    print(X.head())


    #specify the model.
    iowa_model = DecisionTreeRegressor(random_state=1)



    # Fit the model
    iowa_model.fit(X, y)

    predictions = iowa_model.predict(X)
    print(predictions)

if __name__ == '__main__':
    main()