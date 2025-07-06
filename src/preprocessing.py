import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    """
    Extract MFCC features and pad/truncate to a fixed length.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Pad or trim to max_len
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
