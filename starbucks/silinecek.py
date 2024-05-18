import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

starbucks = pd.read_csv("starbucks.csv")
df = starbucks.copy()
df = df.dropna()
dms = pd.get_dummies(df[['Beverage_category', 'Beverage_prep']])
y = df["Calories"]
x_ = df.drop(['Calories', 'Beverage_category', 'Beverage_prep'], axis=1)
x = pd.concat([x_, dms], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

le = LabelEncoder()
y_train_encoder = le.fit_transform(y_train)
y_test_encoder = le.fit_transform(y_test)

# On işleme
print(df.head())
print(df.info())
print(f"Number of Columns :{df.shape[1]}")
print(f"Number of Rows :{df.shape[0]}")
print(pd.DataFrame({
    'count': df.shape[0],
    'Nulls': df.isnull().sum(),
    'nulls%': df.isnull().sum() * 100,
    'cardinality': df.nunique()
}))
