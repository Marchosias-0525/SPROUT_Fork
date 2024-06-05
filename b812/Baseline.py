import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""
Fraction_Insertions,
Avg_Insertion_Length,
Avg_Deletion_Length,
Indel_Diversity,
Fraction_Frameshifts
"""

file_paths = [
    r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train_k2.csv',
    r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train_k3.csv',
    r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train_k4.csv',
    r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train_k5.csv']

k_values = [2, 3, 4, 5]
target_columns = ['Fraction_Insertions', 'Avg_Insertion_Length', 'Avg_Deletion_Length', 'Indel_Diversity', 'Fraction_Frameshifts']

def linearregression_model(file_path, target_column):

    data = pd.read_csv(file_path)
    X = data.iloc[:, :-5]  
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = ...
    train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, r2

results = {}
# Train, evaluate, and save model for each k value and each target variable
for k, file_path in zip(k_values, file_paths):
    for target_column in target_columns:
        mae, mse, r2 = train_evaluate_save_model(file_path, target_column)
        
        if k not in results:
            results[k] = {}
        
        results[k][target_column] = {
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        }
        
# Save results to a CSV file
results_df = pd.DataFrame.from_dict({(k, target): results[k][target] 
                                     for k in results.keys() 
                                     for target in results[k].keys()},
                                    orient='index')
results_df.to_csv(r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\b812\model\model_evaluation_results_separate.csv', index_label=['k', 'target'])

# Display the results
print(results_df)

# Save the model performance
results_df = pd.DataFrame(results).transpose()
results_df.head()
results_df.to_csv(r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\model_evaluation_results.csv', index_label='k')

# Display the results
print(results_df)
performance = pd.DataFrame({'Mean Squared Error': [mse], 'R^2 Score': [r2]})
performance.to_csv('/mnt/data/baseline_model_performance_k2.csv', index=False)
print('Saved baseline model performance to baseline_model_performance_k2.csv')