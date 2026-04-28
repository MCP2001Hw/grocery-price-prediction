import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

k = 3

feature_cols = [
    "Calories (kcal per 100g)",
    "Fat (g per 100g)",
    "Carbohydrate (g per 100g)",
    "Sugars (g per 100g)",
    "Fibre (g per 100g)",
    "Protein (g per 100g)",
    "Salt (g per 100g)",
    "Macro Balance",
]

#load data
df = pd.read_csv("Data.csv", encoding="latin-1")

# make an array of product names
names = df['Name of the product'].astype(str).values

# selects only feature columns and the price drops missing
dt = df[feature_cols + ["Price per 100g"]].apply(pd.to_numeric).dropna()

X = dt.drop(columns=["Price per 100g"]).values
Y = dt["Price per 100g"].values

# split X, Y, and names together so test set names align with preds AI helped here to find and use the train_test_split function
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, Y, names, test_size=1/3, random_state=67
)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# mean abs error and root mean square error
rmse = math.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
print(f"Test samples: {len(y_test)} | RMSE: {rmse} | MAE: {mae}")

# Example predictions: print first 10 test products with name, predicted and actual
print("\n""product,predicted,actual")
for i in range(10):
    print(names_test[i],preds[i],y_test[i])
