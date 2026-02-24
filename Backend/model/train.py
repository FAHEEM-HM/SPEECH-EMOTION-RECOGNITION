import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
LSTM = tf.keras.layers.LSTM



DATASET_PATH = r"C:\Users\TAMIAZ\Documents\SER\TESS"

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
        axis=0
    )
    return mfcc

X = []
y_labels = []

print("📂 Loading dataset...")

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".wav"):
            path = os.path.join(root, file)
            label = file.split("_")[-1].split(".")[0].lower()

            X.append(extract_mfcc(path))
            y_labels.append(label)

print("Total audio files found:", len(X))

X = np.array(X)
X = np.expand_dims(X, -1)

encoder = OneHotEncoder()
y = encoder.fit_transform(np.array(y_labels).reshape(-1, 1)).toarray()

print("🎯 Unique emotions:", encoder.categories_[0])

model = Sequential([
    LSTM(256, input_shape=(40, 1)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(encoder.categories_[0]), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("🚀 Training model...")
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)


model.save("ser_lstm_model.h5")
joblib.dump(encoder, "label_encoder.pkl")

print("✅ Training complete")
print("✅ ser_lstm_model.h5 saved")
print("✅ label_encoder.pkl saved")
