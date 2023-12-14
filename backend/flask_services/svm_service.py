import joblib
import librosa
import numpy as np
import os

# Paths to the model files
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/svm_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')

# Load the trained SVM model and the scaler
svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def extract_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Extract MFCCs from the audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Scale features using standard deviation and mean
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    return mfccs_scaled

def predict_genre(audio_path):
    # Extract features from the audio file
    features = extract_features(audio_path)
    features_scaled = scaler.transform([features])

    # Predict the genre
    predicted_genre_index = svm_model.predict(features_scaled)

    # Convert numpy int64 to Python int before JSON serialization
    predicted_genre_index = int(predicted_genre_index[0])

    # Return the genre name according to the index
    if predicted_genre_index == 0 :
        return "Classical"
    if predicted_genre_index == 1 :
        return "Blues"
    


# Example usage
if __name__ == "__main__":
    # Test the function with a path to an audio file
    test_audio_path = 'path/to/audio.wav'
    print("Predicted Genre:", predict_genre(test_audio_path))

