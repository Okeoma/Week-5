import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the insurance dataset
df = pd.read_csv('insurance.csv')

# Preprocess the dataset (as done in the Streamlit app)
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Define features (X) and target variable (y)
X = df.drop(columns='charges', axis=1)
y = df['charges']

# Train a RandomForestRegressor model
rfr = RandomForestRegressor()
rfr.fit(X, y)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rfr, model_file)
