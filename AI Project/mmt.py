import librosa
import numpy as np

# Note-to-frequency mapping (A4 = 440 Hz)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def hz_to_note(freq):
    """Convert frequency to the nearest musical note."""
    if freq <= 0:
        return None
    # Calculate the number of half steps from A4 (440 Hz)
    half_steps_from_a4 = 12 * np.log2(freq / 440.0)
    # Round to the nearest note
    note_index = int(round(half_steps_from_a4)) % 12
    # Determine the octave
    octave = int((round(half_steps_from_a4) + 9) / 12) + 4
    return f"{NOTE_NAMES[note_index]}{octave}"

# Load the audio file
audio_path = 'sample.wav'
y, sr = librosa.load(audio_path)

# Perform Fourier Transform to find dominant frequency
fft = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(fft), 1/sr)
magnitude = np.abs(fft)
dominant_freq = frequencies[np.argmax(magnitude)]

# Map dominant frequency to a note
note = hz_to_note(dominant_freq)

print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
print(f"Identified Note: {note}")
