import librosa

audio_path = "sample.wav"
try:
    y, sr = librosa.load(audio_path)
    print(f"Audio loaded successfully! Sample rate: {sr}, Data shape: {y.shape}")
except Exception as e:
    print(f"Error loading audio: {e}")
