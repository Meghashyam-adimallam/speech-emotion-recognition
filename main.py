import os
import sys
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Add the 'src' folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset import EmotionDataset

# Emotion code to label
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_ravdess(data_dir):
    file_paths = []
    emotion_labels = []

    for actor in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor)
        wav_files = glob.glob(actor_path + '/*.wav')
        for file in wav_files:
            emotion_code = os.path.basename(file).split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                file_paths.append(file)
                emotion_labels.append(emotion)

    return file_paths, emotion_labels

# Path to audio folder
audio_dir = "data/raw_audio/Audio_Speech_Actors_01-24"
file_paths, string_labels = load_ravdess(audio_dir)

# Encode emotion labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(string_labels)

# Split into train/val/test
X_train, X_test, y_train, y_test = train_test_split(file_paths, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Build PyTorch dataset and loader
train_dataset = EmotionDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Test one batch
if __name__ == "__main__":
    for x_batch, y_batch in train_loader:
        print(f"Input shape: {x_batch.shape}")  # (batch_size, time_steps, features)
        print(f"Labels: {y_batch}")
        break
