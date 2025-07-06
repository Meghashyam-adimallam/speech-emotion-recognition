# train.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Add 'src' to import path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset import EmotionDataset
from model import EmotionBiLSTM

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

# Load and prepare data
audio_dir = "data/raw_audio/Audio_Speech_Actors_01-24"
file_paths, string_labels = load_ravdess(audio_dir)

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(string_labels)

X_train, X_test, y_train, y_test = train_test_split(file_paths, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Create datasets and dataloaders
batch_size = 32
train_dataset = EmotionDataset(X_train, y_train)
val_dataset = EmotionDataset(X_val, y_val)
test_dataset = EmotionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionBiLSTM(input_dim=40, hidden_dim=128, num_layers=2, num_classes=8).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.numpy())

    acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Accuracy: {acc:.4f}")

# Save model
torch.save(model.state_dict(), "emotion_bilstm.pth")
print("✅ Model saved as 'emotion_bilstm.pth'")
