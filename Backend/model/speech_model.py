import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
load_model = tf.keras.models.load_model



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ser_lstm_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")


class SpeechEmotionModel:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.encoder = joblib.load(ENCODER_PATH)

    def extract_mfcc(self, file_path):
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
            axis=0
        )
        return np.expand_dims(mfcc, axis=(0, -1))

    def predict(self, file_path):
        mfcc = self.extract_mfcc(file_path)
        probs = self.model.predict(mfcc)[0]
        idx = np.argmax(probs)

        emotion = self.encoder.categories_[0][idx]
        confidence = float(probs[idx] * 100)

        return emotion, confidence
