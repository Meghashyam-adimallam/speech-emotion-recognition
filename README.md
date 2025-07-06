
# 🎙️ Speech Emotion Recognition with PyTorch + Streamlit

A Deep Learning project that recognizes **human emotions from voice** using **MFCC audio features** and a **BiLSTM model** trained on the RAVDESS dataset. The project includes a full pipeline: preprocessing, training, evaluation, and a real-time Streamlit web app.

---

## 📌 Project Highlights

- 🔊 Input: `.wav` audio files (speech only)
- 🎯 Output: Predicted emotion (e.g., happy, sad, angry, calm...)
- 🧠 Model: Bidirectional LSTM using PyTorch
- 📊 Evaluation: Accuracy, F1-score, and confusion matrix
- 💬 Web App: Built with Streamlit for real-time emotion detection

---

## 🧠 Emotions Detected

The model classifies audio into **8 emotion classes**:

```python
['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
````

---

## 📁 Folder Structure

```
speech-emotion-recognition/
├── app.py                 # Streamlit UI
├── train.py               # Model training script
├── evaluate.py            # Model evaluation on test data
├── emotion_bilstm.pth     # Trained PyTorch model
├── requirements.txt       # Dependencies list
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT license file
├── src/
│   ├── model.py           # BiLSTM model definition
│   ├── dataset.py         # Custom PyTorch Dataset
│   ├── preprocessing.py   # Audio feature extraction (MFCC)
│   └── __init__.py
└── README.md              # This file
```

---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/Meghashyam-adimallam/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Or install manually:

```bash
pip install torch torchaudio librosa matplotlib seaborn scikit-learn streamlit
```

### 3. Train the Model (Optional)

```bash
python train.py
```

### 4. Evaluate on Test Set

```bash
python evaluate.py
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🎙️ Streamlit App Preview

Upload a `.wav` file and see emotion prediction live!

<!-- Optionally add a screenshot -->

<!-- ![App Screenshot](link-to-screenshot.png) -->

---

## 📦 Dataset Used

> **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
> Only the speech audio subset was used.

📥 Download here: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

📝 **Note**: Audio files are not included in this repo due to size limits.
Please download and place them in:

```
data/raw_audio/Audio_Speech_Actors_01-24/
```

---

## 📊 Model Performance

* Validation Accuracy: **\~48–50%**
* Strong detection for: **angry**, **surprised**, **calm**
* Evaluated using confusion matrix + F1 scores

---

## 💡 Future Improvements

* 🎤 Add microphone input support
* 📈 Improve accuracy using spectrograms or CNN layers
* ☁️ Deploy to Hugging Face Spaces or Streamlit Cloud

---

[MIT License](LICENSE)

```

---
