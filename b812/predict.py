import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model_path = r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\b812\model\CNN_Fraction_insertions\cnn_model.h5'

def one_hot_encode(sequence):
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    one_hot = np.zeros((len(sequence), 4))
    for i, nucleotide in enumerate(sequence):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1
    return one_hot


def prepare_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # One-hot encode the DNA sequences
    sequences = data['GuideSeq'].apply(one_hot_encode)
    sequences = np.array(sequences.tolist())
       
    return sequences, data

def load_model_and_predict(model_path, test_file_path, target_column):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    
    # Prepare the test dataset
    X_test, test_data = prepare_dataset(test_file_path)
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Compare the predictions with the actual values
    y_test = test_data[target_column].values  # Assuming we are predicting the target column
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae}, MSE: {mse}, R2: {r2}")
    
    # Add predictions to the test data
    test_data['Predictions'] = y_pred
    
    # Plot Prediction vs Actual
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
    plt.title('Prediction vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(f'NeuralNetwork/SPROUT_Fork/result/CNN_{target_column}/plot/prediction_vs_actual.png')
    plt.show()
    
    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.savefig(f'NeuralNetwork/SPROUT_Fork/result/CNN_{target_column}/plot/residuals.png')
    plt.show()
    
    return test_data,mae, mse, r2

test_file_path = r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\CRISPR.test.csv'
target_column = 'Fraction_Insertions'

predicted_test_data,mae, mse, r2 = load_model_and_predict(model_path, test_file_path, target_column)
results = {
    'CNN': {'MAE': mae, 'MSE': mse, 'R2': r2}
}
results_df = pd.DataFrame(results).transpose()
results_df.to_csv(f'NeuralNetwork/SPROUT_Fork/result/CNN_{target_column}/CNN_model_evaluation_results.csv', index_label='k')
print(results_df)

