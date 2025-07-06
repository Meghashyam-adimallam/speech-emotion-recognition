📄 README.md
markdown
Copy
Edit
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

['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

yaml
Copy
Edit

---

## 📁 Folder Structure

speech-emotion-recognition/
├── app.py # Streamlit UI
├── train.py # Model training script
├── evaluate.py # Model evaluation on test data
├── emotion_bilstm.pth # Trained PyTorch model
├── src/
│ ├── model.py # BiLSTM model definition
│ ├── dataset.py # Custom PyTorch Dataset
│ ├── preprocessing.py # Audio feature extraction (MFCC)
│ └── init.py
└── README.md # This file

yaml
Copy
Edit

---

## 🚀 How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/Meghashyam-adimallam/speech-emotion-recognition.git
cd speech-emotion-recognition
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If you don’t have requirements.txt, install manually:

bash
Copy
Edit
pip install torch torchaudio librosa matplotlib seaborn scikit-learn streamlit
3. Train the Model (Optional)
bash
Copy
Edit
python train.py
4. Evaluate on Test Set
bash
Copy
Edit
python evaluate.py
5. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
🎙️ Streamlit App Preview
Upload a .wav file and see emotion prediction live:

<!-- optional, replace with your image URL -->

📦 Dataset Used
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Only the speech audio subset was used.

Download: https://zenodo.org/record/1188976

📝 Note: Audio files are not included in this repo due to size limits. Please download and place them in data/raw_audio/.

📊 Model Performance
Final Validation Accuracy: ~48–50%

Confusion Matrix shows strengths in detecting angry, surprised, calm

💡 Future Improvements
Add microphone input support

Improve accuracy with spectrograms or data augmentation

Deploy to Hugging Face Spaces or Streamlit Cloud

👤 Author
Meghashyam Adimallam

📜 License
This project is under the MIT License.

yaml
Copy
Edit

---

## ✅ What You Can Do Now:

1. Paste the file into `README.md`
2. Commit + push:
```bash
git add README.md
git commit -m "Add project README"
git push
