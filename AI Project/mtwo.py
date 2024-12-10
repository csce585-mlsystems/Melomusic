import librosa
import numpy as np
import matplotlib.pyplot as plt

# Note-to-frequency mapping (A4 = 440 Hz)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def hz_to_note(freq):
    """Convert frequency to the nearest musical note."""
    if freq <= 0:
        return None
    # Calculate the number of half steps from A4 (440 Hz)
    half_steps_from_a4 = 12 * np.log2(freq / 440.0)
    # Round to the nearest note
    note_index = int(round(half_steps_from_a4)) % 12 -3
    # Determine the octave
    octave = int((round(half_steps_from_a4) + 9) / 12) + 4
    return f"{NOTE_NAMES[note_index]}{octave}"

def analyze_audio_by_second(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration:.2f} seconds")

    # Segment audio into 1-second chunks
    chunk_size = int(sr/2)  # Samples per second
    num_chunks = int(np.ceil(len(y) / chunk_size))

    notes = []
    timestamps = []
        
    for i in range(num_chunks):
            start_sample = i * chunk_size
            end_sample = min((i + 1) * chunk_size, len(y))
            chunk = y[start_sample:end_sample]

            # Perform Fourier Transform to find dominant frequency
            fft = np.fft.fft(chunk)
            frequencies = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)

            # Find the dominant frequency
            positive_freqs = frequencies[:len(frequencies)//2]
            positive_magnitudes = magnitude[:len(magnitude)//2]
            dominant_freq = positive_freqs[np.argmax(positive_magnitudes)]

            # Map dominant frequency to a note
            note = hz_to_note(dominant_freq)
            notes.append(note)
            timestamps.append(i)

            print(f"Half-second {i}: Dominant Frequency = {dominant_freq:.2f} Hz, Note = {note}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, notes, marker='o', linestyle='-', label='Notes')
    plt.xticks(timestamps)
    plt.xlabel('Time (half-seconds)')
    plt.ylabel('Note')
    plt.title('Note vs. Time')
    plt.grid()
    plt.legend()
    plt.show()

# Example usage
audio_path = "sample.wav"
analyze_audio_by_second(audio_path)
