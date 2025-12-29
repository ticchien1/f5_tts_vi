import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Chọn 3 file đại diện của bạn
files = [
    "samples/update_10000_gen.wav",   # Giai đoạn đầu
    "samples/update_400000_gen.wav",  # Giai đoạn giữa
    "samples/update_800000_gen.wav"   # Giai đoạn cuối
]
titles = ["Step 10k", "Step 400k", "Step 800k"]

plt.figure(figsize=(15, 5))

for i, file_path in enumerate(files):
    y, sr = librosa.load(file_path, sr=24000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.subplot(1, 3, i + 1)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.title(titles[i])
    plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.savefig('spectrogram_evolution.png', dpi=300)
plt.show()