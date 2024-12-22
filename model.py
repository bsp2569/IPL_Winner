import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load dataset
df = pd.read_csv('C:/Users/bhaba/Desktop/Projects/Cricket-analysis/IPLScorePredictor-2024/dataset.csv')


# Split features and labels
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)

# Preprocessing pipeline
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')


ra_pipe = Pipeline([
    ('step1', trf),
    ('step2', RandomForestClassifier(random_state=42))
])

# Train the model
ra_pipe.fit(x_train, y_train)

# Evaluate the model
ra_y_pred = ra_pipe.predict(x_test)
print("Accuracy Score:", accuracy_score(y_test, ra_y_pred))

# Save the model
dump(ra_pipe, 'ra_pipe.joblib')
