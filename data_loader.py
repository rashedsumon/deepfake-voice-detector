import os
import glob
import kagglehub
import librosa
import numpy as np
import pandas as pd

def download_dataset():
    """Downloads the latest version of the deepfake voice dataset via kagglehub."""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")
    print("Path to dataset files:", path)
    return path

def extract_features(file_path, max_pad_len=40):
    """Extracts MFCC features from an audio file."""
    try:
        # Load audio file (resample to 16kHz for consistency)
        audio, sample_rate = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
        
        # Extract Mel-Frequency Cepstral Coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Standardize width/length of the audio feature matrix
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
            
        # Flatten into a 1D feature array
        return mfccs.flatten()
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def load_and_preprocess_data():
    """Downloads data, parses folders labeled REAL/FAKE, and extracts features."""
    dataset_path = download_dataset()
    
    features = []
    labels = []
    
    # Looking for 'KAGGLE/AUDIO/REAL' and 'KAGGLE/AUDIO/FAKE' structures inside the download path
    for label_type in ['REAL', 'FAKE']:
        # Match wav or mp3 files recursively
        search_path = os.path.join(dataset_path, '**', label_type, '*.wav')
        file_list = glob.glob(search_path, recursive=True)
        
        # Backup search if nested differently
        if not file_list:
            search_path = os.path.join(dataset_path, '**', label_type.lower(), '*.wav')
            file_list = glob.glob(search_path, recursive=True)

        print(f"Extracting features for {len(file_list)} files in category: {label_type}...")
        
        for file_path in file_list[:300]: # Limit to 300 per class for rapid deployment/training limits
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(label_type) # "REAL" or "FAKE"
                
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Test execution
    X, y = load_and_preprocess_data()
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")