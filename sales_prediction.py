import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("Advertising.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Features (Input)
X = data[['TV', 'Radio', 'Newspaper']]

# Target (Output)
y = data['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nR² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Example prediction
sample = pd.DataFrame(
    [[230.1, 37.8, 69.2]],
    columns=["TV", "Radio", "Newspaper"]
)

predicted_sales = model.predict(sample)

print("\nPredicted Sales:", predicted_sales[0])