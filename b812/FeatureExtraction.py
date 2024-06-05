import pandas as pd
import numpy as np

"""
Fraction_Insertions,
Avg_Insertion_Length,
Avg_Deletion_Length,
Indel_Diversity,
Fraction_Frameshifts
"""

#read data
file_path = r'NeuralNetwork\SPROUT_Fork\data\CRISPR.train.csv'
data = pd.read_csv(file_path)
print(data.isnull().sum())
num_rows, num_columns = data.shape
# Display the count of rows and columns
print(f"Number of rows: {num_rows}",
      f"Number of columns: {num_columns}",sep="\n")

sequence = data['GuideSeq'].tolist()
sequence[:5]
targets = data[['Fraction_Insertions', 'Avg_Insertion_Length', 'Avg_Deletion_Length', 'Indel_Diversity', 'Fraction_Frameshifts']]
# Extract feature
import sys
import os

# Absolute path to the directory containing nacutil.py
absolute_path = r'C:\Users\USER\anaconda3\envs\python39\lib\site-packages\repDNA'
if absolute_path not in sys.path:
    sys.path.append(absolute_path)
    
from repDNA.nac import Kmer
def generate_kmer_features(sequences, targets, k_values, output_prefix):
    for k in k_values:
        kmer = Kmer(k=k)
        features = kmer.make_kmer_vec(sequences)
        features_df = pd.DataFrame(features)
        
        # Concatenate with target variables
        combined_df = pd.concat([features_df, targets.reset_index(drop=True)], axis=1)
        
        # Save to CSV
        output_file = f'{output_prefix}_k{k}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f'Saved k-mer features and targets for k={k} to {output_file}')

# Define k values and output file prefix
k_values = range(2, 7)
output_prefix = r'K:\Programming\python\NeuralNetwork\SPROUT_Fork\data\train'

# Generate and save k-mer features
generate_kmer_features(sequence, targets, k_values, output_prefix)