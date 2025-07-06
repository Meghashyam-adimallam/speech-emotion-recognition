
# 🎙️ Speech Emotion Recognition with PyTorch + Streamlit

A Deep Learning app that detects **emotions from voice recordings** using MFCC audio features and a BiLSTM model. Built with PyTorch for training and Streamlit for a real-time, interactive UI.

---

## 🚀 Features

🔊 Upload `.wav` speech audio files  
🧠 Extract MFCC features using Librosa  
📚 Trained BiLSTM (Bidirectional LSTM) for emotion classification  
📈 Evaluate using confusion matrix, precision, recall, F1  
💬 Streamlit app for real-time voice emotion prediction  
🧼 Clean UI with upload, play, and prediction  
🔐 Dataset not included due to size (RAVDESS must be downloaded manually)

---

## 📁 Project Structure

```

speech-emotion-recognition/
├── app.py                 # Streamlit web app
├── train.py               # Model training
├── evaluate.py            # Test evaluation and confusion matrix
├── emotion\_bilstm.pth     # Trained model weights
├── requirements.txt       # Python dependencies
├── LICENSE
├── .gitignore
├── src/
│   ├── model.py           # BiLSTM architecture
│   ├── dataset.py         # PyTorch dataset and loader
│   ├── preprocessing.py   # MFCC extraction
│   └── **init**.py
└── README.md

````

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Meghashyam-adimallam/speech-emotion-recognition.git
cd speech-emotion-recognition
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchaudio librosa matplotlib seaborn scikit-learn streamlit
```

### 3. Download the dataset (RAVDESS)

Download only the **speech audio subset** from:
📥 [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

Place the extracted folder like this:

```
data/raw_audio/Audio_Speech_Actors_01-24/
```

### 4. Train the model (optional)

```bash
python train.py
```

### 5. Evaluate on test set

```bash
python evaluate.py
```

### 6. Launch the web app

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Uploads `.wav` file and loads audio
2. Extracts MFCCs (Mel-frequency cepstral coefficients)
3. Feeds the features into a trained BiLSTM
4. Predicts one of 8 emotion classes
5. Displays results in a user-friendly web UI

---

## 🧠 Emotion Classes

```python
['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
```

---

## 📎 Example Use Cases

🎧 Build emotion-aware chatbots
🎙️ Analyze call center customer sentiment
🧠 Use in mental health voice analysis
📈 Real-time emotion detection from voice commands

---

## 📊 Model Performance

* Validation Accuracy: \~48–50%
* Strong predictions for: `angry`, `surprised`, `calm`
* Evaluated using accuracy, confusion matrix, and F1-score

---

[MIT License](LICENSE)

---

## 🙌 Acknowledgements

* RAVDESS dataset by Ryerson University
* Librosa for audio feature extraction
* PyTorch for deep learning training
* Streamlit for the UI
* Made with ❤️ by Meghashyam Adimallam

````

---
