from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_services.svm_service import predict_genre
import base64
import os

app = Flask(__name__)
CORS(app)


@app.route('/predict_SVM', methods=['POST'])
def predict():
    # Get the base64 encoded audio file from the request
    content = request.json
    audio_base64 = content['audio']
    audio_bytes = base64.b64decode(audio_base64)
    audio_path = 'temp_audio.wav'

    # Save the temporary audio file
    with open(audio_path, 'wb') as audio_file:
        audio_file.write(audio_bytes)

     # Make the prediction
    genre = predict_genre(audio_path)

    # Clean up the temporary file
    os.remove(audio_path)

    # Return the prediction result
    # Convert NumPy types to Python types before returning JSON response
    return jsonify({'genre': genre})  # Explicit conversion to int

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

