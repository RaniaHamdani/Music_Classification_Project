import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import librosa

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}", e)
        return None
    return mfccs_processed

# Dataset path
data_path = 'data/genres_original'

# Initialize lists for features and labels
features, labels = [], []

# Process each file in each genre directory
for genre in os.listdir(data_path):
    genre_path = os.path.join(data_path, genre)
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        if file_path.endswith('.wav'):
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(genre)

# Convert lists to Numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train the SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(features_scaled, encoded_labels)

# Save the model and scaler
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and label encoder saved to disk.")
