import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib

"""
Fraction_Insertions,
Avg_Insertion_Length,
Avg_Deletion_Length,
Indel_Diversity,
Fraction_Frameshifts
"""

file_path = r'NeuralNetwork\SPROUT_Fork\data\CRISPR.train.csv'
# target_column = 'Fraction_Insertions'
target_column = 'Avg_Insertion_Length'
target_column = 'Avg_Deletion_Length'
target_column = 'Indel_Diversity'
target_column = 'Fraction_Frameshifts'
model_path = f'NeuralNetwork/SPROUT_Fork/b812/model/CNN_{target_column}/cnn_model.h5'

def one_hot_encode(sequence):
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    one_hot = np.zeros((len(sequence), 4))
    for i, nucleotide in enumerate(sequence):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1
    return one_hot


def prepare_dataset(file_path, target_column):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # One-hot encode the DNA sequences
    sequences = data['GuideSeq'].apply(one_hot_encode)
    sequences = np.array(sequences.tolist())
    X = sequences
    y = data[target_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_evaluate_save_cnn_model(file_path, target_column, model_path):
    # Prepare the dataset
    X_train, X_test, y_train, y_test = prepare_dataset(file_path, target_column)
    
    # Build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    print(model)
    # Set up early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    
    # Train the model and capture the learning history
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                        callbacks=[early_stopping, model_checkpoint], verbose=1)
    
    # Predict on the test set
    y_pred = model.predict(X_test).flatten()
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Plot the learning curve
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'NeuralNetwork/SPROUT_Fork/b812/model/CNN_{target_column}/plot/learning_curve.png')
    plt.show()
    
     # Plot Prediction vs Actual
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
    plt.title('Prediction vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(f'NeuralNetwork/SPROUT_Fork/b812/model/CNN_{target_column}/plot/prediction_vs_actual.png')
    plt.show()
    
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.savefig(f'NeuralNetwork/SPROUT_Fork/b812/model/CNN_{target_column}/plot/residuals.png')
    plt.show()
    
    return mae, mse, r2

mae, mse, r2 = train_evaluate_save_cnn_model(file_path, target_column, model_path)
print(f'MAE: {mae}, MSE: {mse}, R2: {r2}')

results = {
    'CNN': {'MAE': mae, 'MSE': mse, 'R2': r2}
}
results_df = pd.DataFrame(results).transpose()
results_df.to_csv(f'NeuralNetwork/SPROUT_Fork/b812/model/CNN_{target_column}/CNN_model_evaluation_results.csv', index_label='k')
print(results_df)