from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import base64
import io
from pydub import AudioSegment
from scipy.signal import spectrogram

app = Flask(__name__)

# Charger le modèle VGG19
vgg19_model = tf.keras.applications.VGG19(include_top=False, input_shape=(224, 224, 3))

def audio_to_spectrogram(audio_bytes):
    """
    Convertit les données audio en un spectrogramme adapté pour VGG19.
    """
    # Conversion du fichier audio en spectrogramme
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    # Extraire les données brutes du fichier audio
    data = np.array(audio.get_array_of_samples())
    # Génération du spectrogramme
    freqs, times, spec = spectrogram(data, fs=audio.frame_rate)
    
    # Normalisation et redimensionnement pour VGG19
    spec = np.log(spec + 1e-6)
    spec = np.expand_dims(spec, axis=-1)
    spec = np.tile(spec, (1, 1, 3))  # Duplication des canaux pour correspondre à l'input RGB
    spec = tf.image.resize(spec, (224, 224))
    spec = np.expand_dims(spec, axis=0)

    return spec

@app.route('/vgg19_service', methods=['POST'])
def vgg19_service():
    data = request.json['wav_music']
    wav_data = base64.b64decode(data)
    
    # Conversion en spectrogramme
    spec = audio_to_spectrogram(wav_data)
    
    # Prédiction avec le modèle VGG19
    predictions = vgg19_model.predict(spec)
    # Traitement supplémentaire pour déterminer le genre
    # (Cela dépend de la manière dont vous formez ou adaptez le modèle à la classification des genres musicaux)

    # Envoie de la réponse
    return jsonify({'genre': predictions})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
