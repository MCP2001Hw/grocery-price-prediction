# Import necessary libraries
import numpy, pandas
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
data = pandas.read_csv("Data.csv", header=0, encoding='latin-1')

# Shuffle and split
seed = numpy.random.randint(0, 1_000_000)
train = data.sample(frac=2/3, random_state=seed)
test = data.drop(train.index)

# Save to new files
train.to_csv("train_dtree.csv", index=False)
test.to_csv("test_dtree.csv", index=False)

# Import the training data sheet
train_data = pandas.read_csv("train_dtree.csv", header=0, encoding='latin-1')

# Feature columns
feature_columns = [
                    # "weight of the product",
                    "Calories (kcal per 100g)",
                    "Fat (g per 100g)",
                    "Carbohydrate (g per 100g)",
                    "Sugars (g per 100g)",
                    "Fibre (g per 100g)",
                    "Protein (g per 100g)",
                    "Salt (g per 100g)",
                    "Macro Balance"
                   ]

# test_column = "Price of the product"
test_column = "Price per 100g"

# Split out the input data
train_input_data = train_data[feature_columns].to_numpy(dtype=float)

# Split out the correct data
train_correct_data = train_data[test_column].to_numpy(dtype=float)

def train(max_depth, train_input_data, train_correct_data):
    # Create and train the decision tree model
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42, min_samples_split=5, min_samples_leaf=2)
    model.fit(train_input_data, train_correct_data)
    return model

def test(model, feature_columns, test_column):
    # Import the testing data sheet
    test_data = pandas.read_csv("test_dtree.csv", header=0, encoding='latin-1')

    # Split the input data out
    test_input_data = test_data[feature_columns].to_numpy(dtype=float)

    true_outcome = test_data[test_column].to_numpy(dtype=float)

    # Get predicted outcome
    predicted_outcome = model.predict(test_input_data)
    return predicted_outcome, true_outcome

def get_product_features(product_name, feature_columns):
    # Load the dataset to find the product features
    all_data = pandas.read_csv("Data.csv", header=0, encoding='latin-1')
    product_data = all_data[all_data['Name of the product'] == product_name]
    if product_data.empty:
        return None
    return product_data[feature_columns].to_numpy(dtype=float)

def main(max_depth):
    print(f"\nTraining Decision Tree with max depth: {max_depth}")

    # Load the dataset for later use
    all_data = pandas.read_csv("Data.csv", header=0, encoding='latin-1')

    # Train the model
    model = train(max_depth, train_input_data, train_correct_data)

    print("="*50)
    print(f"Feature importances:")
    for i, feature in enumerate(feature_columns):
        print(f"-{feature}: {model.feature_importances_[i]:.4f}")
    print("="*50)
    print(f"Looking for: {test_column}")
    print("="*50)

    # Test the model
    predicted_outcome, true_outcome = test(model, feature_columns, test_column)
    mean_square_error = numpy.mean((true_outcome - predicted_outcome)**2)
    root_mean_square_error = numpy.sqrt(mean_square_error)
    mean_absolute_err = numpy.mean(numpy.abs(true_outcome - predicted_outcome))

    percentage_errors = numpy.abs((true_outcome - predicted_outcome) / true_outcome) * 100
    percentage_errors = percentage_errors[numpy.isfinite(percentage_errors)]
    mean_percentage_error = numpy.mean(percentage_errors)

    print(f"Mean Square Error: {mean_square_error:.4f}")
    print(f"Root Mean Square Error: {root_mean_square_error:.4f}")
    print(f"Mean Absolute Error: {mean_absolute_err:.4f}")
    print(f"Mean Absolute Percentage Error: {mean_percentage_error:.2f}%")
    print("="*50)

    while True:
        # Input product name for prediction
        product_name = input("Enter the Name of the product or Ctrl+C to quit: ")
        product_features = get_product_features(product_name, feature_columns)
        if product_features is not None:
            predicted_price = model.predict(product_features)
            print(f"Predicted price for '{product_name}': {predicted_price[0]:.4f}")

            # Retrieve actual price for comparison
            actual_price = all_data.loc[all_data['Name of the product'] == product_name, test_column].values[0]
            print(f"Actual price for '{product_name}': {actual_price:.4f}")
        else:
            print(f"Product '{product_name}' not found.")

main(max_depth=10)
