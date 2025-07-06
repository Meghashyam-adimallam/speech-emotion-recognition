# evaluate.py

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add 'src' folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset import EmotionDataset
from model import EmotionBiLSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import glob

from src.preprocessing import extract_mfcc

# Emotion map
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
            basename = os.path.basename(file)
            emotion_code = basename.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                file_paths.append(file)
                emotion_labels.append(emotion)
    return file_paths, emotion_labels

# Load and prepare test data
audio_dir = "data/raw_audio/Audio_Speech_Actors_01-24"
file_paths, string_labels = load_ravdess(audio_dir)

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(string_labels)

# Final test split (same as train.py)
_, X_test, _, y_test = train_test_split(file_paths, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

test_dataset = EmotionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionBiLSTM(input_dim=40, hidden_dim=128, num_layers=2, num_classes=8)
model.load_state_dict(torch.load("emotion_bilstm.pth"))
model.to(device)
model.eval()

# Evaluate on test data
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Print results
print("\n✅ Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
