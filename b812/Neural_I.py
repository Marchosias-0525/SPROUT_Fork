import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

file_paths = [
    r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train_k2.csv',
    r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train_k3.csv',
    r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train_k4.csv',
    r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train_k5.csv']
k_values = [2, 3, 4, 5]
target_columns = ['Fraction_Insertions', 'Avg_Insertion_Length', 'Avg_Deletion_Length', 'Indel_Diversity', 'Fraction_Frameshifts']


def build_nn_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer with 1 neuron for regression
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

"""
def build_nn_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    # Use Adam optimizer with a custom learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def build_nn_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dense(1))
    
    # Use Adam optimizer with a custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model
"""


# Function to train, evaluate, and save a model for a specific target variable
def train_evaluate_save_nn_model(file_path, target_column, model_path,  k_value):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, :-5] 
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build the neural network model
    model = build_nn_model(X_train.shape[1])
    print(model.summary())
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    
    # Train
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                        callbacks=[early_stopping, model_checkpoint], verbose=1)
    
    # history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    
    # Predict 
    y_pred = model.predict(X_test).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save the model
    model.save(model_path)
    
    # Plot the learning curve
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'Learning Curve for k={k_value}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'NeuralNetwork/SPROUT_Fork/b812/model/Fraction_Insertions/plot/learning_curve_k{k_value}.png')
    plt.show()
    
     # Plot Prediction vs Actual
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
    plt.title(f'Prediction vs Actual for k={k_value}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(f'NeuralNetwork/SPROUT_Fork/b812/model/Fraction_Insertions/plot/prediction_vs_actual_k{k_value}.png')
    plt.show()
    
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residuals for k={k_value}')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.savefig(f'NeuralNetwork/SPROUT_Fork/b812/model/Fraction_Insertions/plot/residuals_k{k_value}.png')
    plt.show()
    
    return mae, mse, r2


##

    
# Build the neural network model

results = {}
target_column='Fraction_Insertions'
# Train, evaluate, and save model for each k value and target variable
for k, file_path in zip(k_values, file_paths):
    model_path = f'NeuralNetwork/SPROUT_Fork/b812/model/Fraction_Insertions/nn_model_k{k}_{target_column}.h5'
    mae, mse, r2 = train_evaluate_save_nn_model(file_path, target_column, model_path, k)
    
    results[k] = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }

results_df = pd.DataFrame(results).transpose()
results_df.to_csv('NeuralNetwork/SPROUT_Fork/b812/model/Fraction_Insertions/nn_model_evaluation_results_k2_k5.csv', index_label='k')

# Display the results
print(results_df)