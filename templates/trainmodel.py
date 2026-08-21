import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv('cleanedcar.csv')

X = df[['year','kms_driven','name','company','fuel_type']]
Y = df['Price']

ohe = OneHotEncoder(handle_unknown='ignore')
column_trans = ColumnTransformer(
    transformers=[
        ('cat', ohe, ['name','company','fuel_type'])
    ],
    remainder='passthrough'
)
pipeline = Pipeline([
    ('preprocessor', column_trans),
    ('model', LinearRegression())
])
pipeline.fit(X, Y)
pickle.dump(pipeline, open("model.pkl", "wb"))