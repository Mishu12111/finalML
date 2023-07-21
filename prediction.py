import os
import shutil
import librosa
import pandas as pd
import sys
import pickle
import json
from caracteristici import extract_audio_features

# Se incarca modelele de machine learning
model_rf = pickle.load(open('csharp_rf.pkl', 'rb'))
model_lr = pickle.load(open('csharp_lr.pkl', 'rb'))
# Fisierele de input/output
input_dir = 'data/test_wav'
output_dir1 = 'manele1'
output_dir0 = 'manele0'

import numpy as np

# Se incarca file CSV deja existenta numita 'features.csv'
df = pd.read_csv('features.csv')

# Se face o lista cu denumirile caracteristicilor din fila CSV
feature_names = list(df.columns)
feature_names.remove('manele')  # Se elimina coloana 'manele'

# Print the length of the DataFrame
print("Length of DataFrame:", len(df))
input_dir_length = len(os.listdir(input_dir))
print("Length of input directory:", input_dir_length)
# Loop through all the files in the input directory
for i, file in enumerate(os.listdir(input_dir)):
    # Check if the file is a supported audio format
    if file.endswith('.wav'):
        # Load the audio file and extract its features
        filename = os.path.join(input_dir, file)
        signal, sr = librosa.load(filename)
        extracted_features = pd.DataFrame(extract_audio_features(signal, sr)).mean(axis=1).values

        # Create a DataFrame with feature names and values
        features_df = pd.DataFrame([extracted_features], columns=feature_names)
        print(features_df)

        # Make a prediction using the trained machine learning model
        prediction_lr = model_lr.predict(features_df)[0]
        prediction_rf = model_rf.predict(features_df)[0]
        print("LINEAR REGRESSION:")
        print(prediction_lr)
        print("RANDOM FOREST REGRESSION:")
        print(prediction_rf)

        # If the predicted label is 'manele', move the file to the output directory
        if prediction_rf > 0.5:
            shutil.move(os.path.join(input_dir, file), os.path.join(output_dir1, file))
            print(f"Moved {file} to {output_dir1}")
            row_index = len(df['manele']) + i
            print("Calculated row index:", row_index)

            df.loc[row_index-input_dir_length, 'manele'] = 1

        else:
            shutil.move(os.path.join(input_dir, file), os.path.join(output_dir0, file))
            print(f"{file} ESTE adecvata.")
            row_index = len(df['manele']) +i
            print("Calculated row index:", row_index)

            df.loc[row_index-input_dir_length, 'manele'] = 0



# Save the updated DataFrame to the CSV file
df.to_csv('features.csv', index=False)
