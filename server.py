import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset and train the model
df = pd.read_csv('Iris.csv')
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC()
model.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Create DataFrame from the input data with the appropriate column names
    input_data = pd.DataFrame([data], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    # Model prediction
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
