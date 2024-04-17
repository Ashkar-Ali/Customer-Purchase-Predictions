from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('purchase_history.csv')

# Separate features (X) and target variable (y)
X = data[['Gender', 'Age', 'Salary', 'Price', 'Product Category']]
y = data['Purchased']

# Perform one-hot encoding for categorical variables (if needed)
X = pd.get_dummies(X, columns=['Gender', 'Product Category'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        age = float(request.form['age'])
        salary = float(request.form['salary'])
        price = float(request.form['price'])
        product_category = request.form['product_category']

        # Create a DataFrame with input data
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Salary': [salary],
            'Price': [price],
            'Product Category': [product_category]
        })

        # Perform one-hot encoding with all possible categories
        input_data = pd.get_dummies(input_data, columns=['Gender', 'Product Category'])

        # Reindex input data to match the features used during training
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Standardize the input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        if prediction[0] == 1:
            result = "Customer is likely to purchase the product."
        else:
            result = "Customer is not likely to purchase the product."

        return render_template('result.html', result=result, probability=probability[0][1])

if __name__ == '__main__':
    app.run(debug=True)
