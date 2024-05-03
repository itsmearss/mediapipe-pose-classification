import pandas as pd
import pickle # Object serialization.

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 

def load_dataset(csv_data):
    # Load dataset from CSV file.
    df = pd.read_csv(csv_data)

    # Separate features and target value.
    features = df.drop('class', axis=1) # Features, drop the column 'class'.
    target_value = df['class']          # Target value.

    # Split dataset into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.3, random_state=1234)

    return x_train, x_test, y_train, y_test

def evaluate_model(fit_models, x_test, y_test):
    print('\nEvaluate model accuracy:')
    # Evaluate and Serialize Model.
    for key_algo, value_pipeline in fit_models.items():
        # Predict using the trained model.
        yhat = value_pipeline.predict(x_test)
        # Calculate accuracy.
        accuracy = accuracy_score(y_test, yhat) * 100
        print(f'Classify algorithm: {key_algo}, Accuracy: {accuracy:.2f}%')
        print('-----------------------------------------------')
        # Generate classification report.
        report = classification_report(y_test, yhat)
        print(f"Classification Report for {key_algo}:")
        print(report)

if __name__ == '__main__':
    
    dataset_csv_file = 'datasets/coords_dataset2.csv'
    model_weights = 'model/model-1.pkl'

    # Load dataset.
    x_train, x_test, y_train, y_test = load_dataset(csv_data=dataset_csv_file)
    
    # Define pipeline with Random Forest Classifier.
    pipelines = {
        'rf' : make_pipeline(StandardScaler(), RandomForestClassifier()),
    }

    fit_models = {}
    print('Model is Training ....')
    # Train models and store them in fit_models dictionary.
    for key_algo, value_pipeline in pipelines.items():
        model = value_pipeline.fit(x_train, y_train)
        fit_models[key_algo] = model
    print('Training done.')

    # Using trained model to predict on test data.
    rc_predict = fit_models['rf'].predict(x_test)
    print(f'\nPredict 5 datas: {rc_predict[0:5]}')

    # Save model weights.
    with open(model_weights, 'wb') as f:
        pickle.dump(fit_models['rf'], f)
    print('\nSave model done.')
    
    # Evaluate model accuracy.
    evaluate_model(fit_models, x_test, y_test)
