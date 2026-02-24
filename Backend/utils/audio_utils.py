import tempfile
import librosa
import soundfile as sf
import os


def convert_to_wav(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.file.read())
        temp_path = temp.name

    y, sr = librosa.load(temp_path, sr=None)
    wav_path = temp_path + ".wav"
    sf.write(wav_path, y, sr)

    os.remove(temp_path)
    return wav_path
