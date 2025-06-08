import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

if __name__ == "__main__":

        df = pd.read_csv('Problem 1—quality.csv')
        df['label_scaled'] = df['label'].map({'G': 1, 'B': 0})

        # Separate features (X) and target (y)
        X = df[['num_words', 'num_characters', 'num_misspelled', 'bin_end_qmark',
                'num_interrogative', 'bin_start_small', 'num_sentences', 'num_punctuations']]
        y = df['label_scaled']
        X = sm.add_constant(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = sm.Logit(y_train, X_train)
        result = model.fit(maxiter=200)
        y_pred_prob = result.predict(X_test)  # Probabilities for each class

        # Get the predicted class by choosing the class with the highest probability
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Calculate accuracy
        accuracy_logit = accuracy_score(y_test, y_pred)

        print(f"Accuracy of Logit Model predicting Program: {accuracy_logit:.2f}")