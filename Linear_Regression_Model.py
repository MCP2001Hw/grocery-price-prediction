# Import numpy for math and pandas for csv management
import numpy, pandas

# Load dataset
data = pandas.read_csv("Data.csv", header=0, encoding='latin-1')

# Shuffle and split
seed = numpy.random.randint(0, 1_000_000)
train = data.sample(frac=2/3, random_state=seed)
test = data.drop(train.index)

# Save to new files
train.to_csv("Train.csv", index=False)
test.to_csv("Test.csv", index=False)

# import the training data sheet
train_data = pandas.read_csv("Train.csv", header=0, encoding='latin-1')

# Feature columns
feature_columns = [
                    # "weight of the product", 
                    "Calories (kcal per 100g)", 
                    "Fat (g per 100g)", 
                    "Carbohydrate (g per 100g)", 
                    "Sugars (g per 100g)", 
                    "Fibre (g per 100g)", 
                    "Protein (g per 100g)", 
                    "Salt (g per 100g)"
                   ]

# test_column = "Price of the product"
test_column = "Price per 100g"

# Split out the input datas
train_input_data = train_data[feature_columns].to_numpy(dtype=float)

# Split out the correct data
train_correct_data = train_data[test_column].to_numpy(dtype=float)

# Normalise the input data
train_mean = numpy.mean(train_input_data, axis=0)
train_std = numpy.std(train_input_data, axis=0)
train_std[train_std == 0] = 1.0
train_input_data = (train_input_data - train_mean) / train_std

# Get the numbers of rows and columns
_, train_numbers_of_feature = train_input_data.shape

def initialise_formula(train_numbers_of_feature):
    # Initialise the weight of each headers / columns
    weight = numpy.zeros(train_numbers_of_feature, dtype=float)
    # Initialise the bias
    bias = 0.0
    return weight, bias

def predict_data(input_data, weight, bias):
    # Use formula W * X + B to predict a number base on input
    predicted_data = numpy.dot(input_data, weight) + bias
    return predicted_data

def run_epoch(learning_rate, weight, bias, epoch, train_input_data, train_correct_data):
    train_numbers_of_data = train_input_data.shape[0]
    # Initialise Mean Square Error
    prev_mse = float('inf')
    
    # Loop over "epoch" amount of times
    for i in range(epoch):
        # Calculate predictions for all input data
        predictions = predict_data(train_input_data, weight, bias)
        
        # Calculate errors for all samples
        errors = train_correct_data - predictions
        
        # Calculate epoch loss
        epoch_loss = numpy.sum(errors ** 2)
        mse = epoch_loss / train_numbers_of_data
        
        # Update weight: w = w + η * (1/n) * Σ(y−ŷ)x
        weight = weight + learning_rate * (1 / train_numbers_of_data) * numpy.dot(train_input_data.T, errors)
        # Update bias: b = b + η * (1/n) * Σ(y−ŷ)
        bias = bias + learning_rate * (1 / train_numbers_of_data) * numpy.sum(errors)
        
        # Print details every 200 epochs
        if (i + 1) % 200 == 0:
            weight_norm = numpy.linalg.norm(weight) 
            print(f"Epoch {i+1}/{epoch}, MSE: {mse:.4f}, Weight norm: {weight_norm:.4f}, Bias: {bias:.4f}")
        
        # Check for convergence
        if abs(prev_mse - mse) < 0.00001:
            print(f"\nConverged at epoch {i+1}")
            break
        
        prev_mse = mse
    
    return weight, bias

def train(epoch, learning_rate, train_numbers_of_feature):
    weight, bias = initialise_formula(train_numbers_of_feature)
    weight, bias = run_epoch(learning_rate, weight, bias, epoch, train_input_data, train_correct_data)
    return weight, bias

def test(weight, bias, train_mean, train_std):
    # Import the testing data sheet
    test_data = pandas.read_csv("Test.csv", header=0, encoding='latin-1')
    
    # Split the input data out
    test_input_data = test_data[feature_columns].to_numpy(dtype=float)
    
    true_outcome = test_data[test_column].to_numpy(dtype=float)

    # Normalise using training data statistics
    test_input_data = (test_input_data - train_mean) / train_std
    
    # Get predicted outcome
    predicted_outcome = predict_data(test_input_data, weight, bias)
    return predicted_outcome, true_outcome

def get_product_features(product_name):
    # Load the dataset to find the product features
    all_data = pandas.read_csv("Data.csv", header=0, encoding='latin-1')
    product_data = all_data[all_data['Name of the product'] == product_name]
    if product_data.empty:
        return None
    return product_data[feature_columns].to_numpy(dtype=float)

def main(training_amount, learning_rate, train_numbers_of_feature):
    print(f"\nTraining with up to {training_amount} epochs and learning rate {learning_rate}")
    
    # Load the dataset for later use
    all_data = pandas.read_csv("Data.csv", header=0, encoding='latin-1')
    
    # Trains the model
    weight, bias = train(training_amount, learning_rate, train_numbers_of_feature)
    
    print("="*50)
    print(f"Bias: {bias:.4f}")
    print("="*50)
    print(f"Weights by feature:")
    for i, feature in enumerate(feature_columns):
        print(f"-{feature}: {weight[i]:.4f}")
    print("="*50)
    print(f"Looking for: {test_column}")
    print("="*50)
    
    # Test the model
    predicted_outcome, true_outcome = test(weight, bias, train_mean, train_std)
    mean_square_error = numpy.mean((true_outcome - predicted_outcome)**2)

    percentage_errors = numpy.abs((true_outcome - predicted_outcome) / true_outcome) * 100
    percentage_errors = percentage_errors[numpy.isfinite(percentage_errors)]
    
    print(f"Mean Square Error: {mean_square_error:.4f}")
    print(f"Root Mean Square Error: {numpy.sqrt(mean_square_error):.4f}")


    while True:
    # Input product name for prediction
        product_name = input("Enter the Name of the product or Ctrl+C to quit: ")
        product_features = get_product_features(product_name)
        if product_features is not None:
            # Normalise the product features using training data statistics
            product_features_normalised = (product_features - train_mean) / train_std
            predicted_price = predict_data(product_features_normalised, weight, bias)
            print(f"Predicted price for '{product_name}': {predicted_price[0]:.4f}")
        
            # Retrieve actual price for comparison
            actual_price = all_data.loc[all_data['Name of the product'] == product_name, test_column].values[0]
            print(f"Actual price for '{product_name}': {actual_price:.4f}")
        else:
            print(f"Product '{product_name}' not found.")
            # Retrieve actual price for comparison
            actual_price = all_data.loc[all_data['Name of the product'] == product_name, test_column].values[0]
            print(f"Actual price for '{product_name}': {actual_price:.4f}")


main(1000000, 0.0004, train_numbers_of_feature)