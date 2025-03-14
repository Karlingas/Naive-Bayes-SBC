from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import copy

# Split the data into X and y, and then into train and test sets
def train_test(df):
    # Split the data into X and y
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

    return X_train, X_test, y_train, y_test

# Split the data into X and y, and then into train and test sets, discretizing the data into n_bins
def train_test_discretize(df, n_bins=3):
    # Split the data into X and y
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_train = discretizer.fit_transform(X_train)
    X_test = discretizer.transform(X_test)

    return X_train, X_test, y_train, y_test


# Compare two scikit-learn models on a database, discretizing the data into n_bins
def compare_two_models_KBinsDiscretizer(data, model1, model2):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test(data)

    # Discretize the data in bins from 2 to 10. Then train the models and predict the test set, printing at the end the accuracy of each model
    for i in range(2, 11):
        # Discretize the data
        discretizer = KBinsDiscretizer(n_bins=i, encode='ordinal', strategy='uniform')
        X_train_temp = discretizer.fit_transform(X_train)
        X_test_temp = discretizer.transform(X_test)

        # Train the models
        model1.fit(X_train_temp, y_train)
        model2.fit(X_train_temp, y_train)
        
        # Predict the test set
        y_pred1 = model1.predict(X_test_temp)
        y_pred2 = model2.predict(X_test_temp)

        # Print the accuracy of each model
        print("Accuracy of model",model1.__class__.__name__,"with",i,"bins:",accuracy_score(y_test, y_pred1),
              "\tAccuracy of model",model2.__class__.__name__,"with",i,"bins:",accuracy_score(y_test, y_pred2))

def see_probability_tables(model):
    if model.__class__.__name__ == "CategoricalNB":
        print("P(Y):\n", np.exp(model.class_log_prior_), "\n")
        
        for i in range(len(model.feature_log_prob_)):
            print("P(X{}|Y):".format(i))
            # Obtener las probabilidades condicionales para la característica i
            feature_probs = np.exp(model.feature_log_prob_[i])  # Convertir de log-prob a prob
            for cls_idx, cls_probs in enumerate(feature_probs):
                print(f"{model.classes_[cls_idx]}:\t{np.round(cls_probs, 4)}")
            print()
    
    elif model.__class__.__name__ == "GaussianNB":
        print("P(Y):\n", model.class_prior_, "\n")
        
        n_features = model.theta_.shape[1]
        for i in range(n_features):
            print("P(X{}|Y):".format(i))
            for cls_idx in range(len(model.classes_)):
                mean = model.theta_[cls_idx, i]
                var = model.var_[cls_idx, i]
                print(f"{model.classes_[cls_idx]}:\tμ = {mean:.4f}, σ² = {var:.4f}")
            print()
            
    elif model.__class__.__name__ == "NB":
        print("P(Y):\n", model.class_prob_, "\n")
        
        for i, table in enumerate(model.tables_):
            if model.is_categorical_[i]:  # Característica categórica
                print("P(X{}|Y):".format(i))
                for cls, probs in table.items():
                    print(f"{cls}:\t{np.round(probs, 4)}")
                print()
            else:  # Característica gaussiana
                print("P(X{}|Y):".format(i))
                for cls, (mean, var) in table.items():
                    print(f"{cls}:\tμ = {mean:.4f}, σ² = {var:.4f}")
                print()