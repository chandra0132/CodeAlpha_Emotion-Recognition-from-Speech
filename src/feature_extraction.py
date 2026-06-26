import os
import glob
import numpy as np
# pyrefly: ignore [missing-import]
import librosa
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Map RAVDESS emotion numbers to string labels and integer codes
EMOTION_MAP = {
    "01": ("neutral", 0),
    "02": ("calm", 1),
    "03": ("happy", 2),
    "04": ("sad", 3),
    "05": ("angry", 4),
    "06": ("fearful", 5),
    "07": ("disgust", 6),
    "08": ("surprised", 7)
}

def augment_audio(y, sr=22050):
    """
    Applies random white noise, pitch shifting, and time stretching to raw audio.
    """
    augmented_signals = []
    
    # 1. White Noise Injection
    noise_amp = 0.005 * np.random.uniform() * np.max(y) if np.max(y) > 0 else 0.001
    noise = np.random.normal(size=y.shape[0])
    augmented_signals.append(y + noise_amp * noise)
    
    # 2. Pitch Shifting (Up and down by 2 semitones)
    try:
        augmented_signals.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2))
        augmented_signals.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2))
    except Exception:
        pass
        
    # 3. Time Stretching (Speed up)
    try:
        y_stretch = librosa.effects.time_stretch(y=y, rate=1.15)
        # Pad or truncate to keep exact length matching input
        target_len = len(y)
        if len(y_stretch) > target_len:
            y_stretch = y_stretch[:target_len]
        else:
            y_stretch = np.pad(y_stretch, (0, target_len - len(y_stretch)), 'constant')
        augmented_signals.append(y_stretch)
    except Exception:
        pass
        
    return augmented_signals

def extract_audio_features(file_path_or_y, sr=22050, duration=3.0, max_pad_len=130):
    """
    Extracts MFCC, Chroma, Mel Spectrogram, Spectral Contrast, RMS, and ZCR features.
    Can accept either a file path (string) or a pre-loaded audio array.
    
    Returns:
        features_2d: 2D numpy array of shape (max_pad_len, 189) containing temporal sequences.
        features_1d: 1D numpy array of shape (189,) containing the average of features over time.
    """
    try:
        if isinstance(file_path_or_y, str):
            # Load audio file
            y, sample_rate = librosa.load(file_path_or_y, sr=sr)
        else:
            y = file_path_or_y
            
        # Trim silence at the start and end of the audio file
        y, _ = librosa.effects.trim(y)
        
        # Standardize duration
        target_samples = int(duration * sr)
        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
            
        # 1. Extract MFCC (40 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # 2. Extract Chroma (12 coefficients)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # 3. Extract Mel Spectrogram (128 coefficients)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # 4. Extract Spectral Contrast (7 coefficients)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # 5. Extract Root-Mean-Square energy (1 coefficient)
        rms = librosa.feature.rms(y=y)
        
        # 6. Extract Zero-Crossing Rate (1 coefficient)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        # Combine all features along the frequency/coefficient axis
        # Shape: (40 + 12 + 128 + 7 + 1 + 1, T) = (189, T)
        features = np.vstack([mfcc, chroma, mel_db, contrast, rms, zcr])
        features = features.T # Shape: (T, 189)
        
        # Pad or truncate along the time axis
        T = features.shape[0]
        if T < max_pad_len:
            pad_width = max_pad_len - T
            features_2d = np.pad(features, ((0, pad_width), (0, 0)), 'constant')
        else:
            features_2d = features[:max_pad_len, :]
            
        # Get time-averaged feature vector (1D)
        features_1d = np.mean(features_2d, axis=0)
        
        return features_2d, features_1d
    except Exception as e:
        print(f"[WARNING] Error processing audio input: {e}")
        return None, None

def load_dataset_features(data_dir="data", test_size=0.2, random_state=42):
    """
    Parses the dataset directory, splits into train/test, augments the training split,
    extracts features (189-dim), and returns scaled splits.
    """
    print("[INFO] Starting feature extraction pipeline with data augmentation...")
    
    # Path pattern: data/Actor_*/*.wav
    wav_files = glob.glob(os.path.join(data_dir, "Actor_*", "*.wav"))
    
    if not wav_files:
        raise ValueError(f"No WAV files found in data directory: {data_dir}. Did you run the download script?")
        
    print(f"[INFO] Found {len(wav_files)} files. Splitting into train/test...")
    
    labels = []
    valid_files = []
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in EMOTION_MAP:
                labels.append(EMOTION_MAP[emotion_code][1])
                valid_files.append(file_path)
                
    labels = np.array(labels)
    
    # Split file paths first to completely separate test samples before augmentation
    file_train, file_test, _, _ = train_test_split(
        valid_files, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # 1. Process Train Set with Data Augmentation
    print(f"[INFO] Extracting features and applying augmentation on {len(file_train)} training files...")
    features_list_2d_train = []
    features_list_1d_train = []
    y_train_final = []
    
    for idx, file_path in enumerate(file_train):
        if (idx + 1) % 150 == 0:
            print(f"Processed {idx + 1}/{len(file_train)} training files...")
            
        emotion_code = os.path.basename(file_path).split('-')[2]
        label = EMOTION_MAP[emotion_code][1]
        
        # Original clip
        f_2d, f_1d = extract_audio_features(file_path)
        if f_2d is not None:
            features_list_2d_train.append(f_2d)
            features_list_1d_train.append(f_1d)
            y_train_final.append(label)
            
            # Augmented clips
            try:
                y, sr = librosa.load(file_path, sr=22050)
                augmented_audios = augment_audio(y, sr=22050)
                for aug_y in augmented_audios:
                    aug_f_2d, aug_f_1d = extract_audio_features(aug_y, sr=22050)
                    if aug_f_2d is not None:
                        features_list_2d_train.append(aug_f_2d)
                        features_list_1d_train.append(aug_f_1d)
                        y_train_final.append(label)
            except Exception as e:
                pass
                
    X_train_2d_raw = np.array(features_list_2d_train)
    X_train_1d_raw = np.array(features_list_1d_train)
    y_train = np.array(y_train_final)
    
    # 2. Process Test Set (Clean - No Augmentation)
    print(f"[INFO] Extracting features for {len(file_test)} testing files...")
    features_list_2d_test = []
    features_list_1d_test = []
    y_test_final = []
    
    for file_path in file_test:
        emotion_code = os.path.basename(file_path).split('-')[2]
        label = EMOTION_MAP[emotion_code][1]
        
        f_2d, f_1d = extract_audio_features(file_path)
        if f_2d is not None:
            features_list_2d_test.append(f_2d)
            features_list_1d_test.append(f_1d)
            y_test_final.append(label)
            
    X_test_2d_raw = np.array(features_list_2d_test)
    X_test_1d_raw = np.array(features_list_1d_test)
    y_test = np.array(y_test_final)
    
    print(f"[INFO] Train samples after augmentation: {len(y_train)}")
    print(f"[INFO] Test samples (clean): {len(y_test)}")
    
    # 3. Standardize 1D features
    scaler_1d = StandardScaler()
    X_train_1d = scaler_1d.fit_transform(X_train_1d_raw)
    X_test_1d = scaler_1d.transform(X_test_1d_raw)
    
    # 4. Standardize 2D features
    num_train, time_steps, num_features = X_train_2d_raw.shape
    X_train_2d_flat = X_train_2d_raw.reshape(-1, num_features)
    scaler_2d = StandardScaler()
    X_train_2d_flat_scaled = scaler_2d.fit_transform(X_train_2d_flat)
    X_train_2d = X_train_2d_flat_scaled.reshape(num_train, time_steps, num_features)
    
    num_test = X_test_2d_raw.shape[0]
    X_test_2d_flat = X_test_2d_raw.reshape(-1, num_features)
    X_test_2d_flat_scaled = scaler_2d.transform(X_test_2d_flat)
    X_test_2d = X_test_2d_flat_scaled.reshape(num_test, time_steps, num_features)
    
    return {
        "X_train_2d": X_train_2d,
        "X_test_2d": X_test_2d,
        "X_train_1d": X_train_1d,
        "X_test_1d": X_test_1d,
        "y_train": y_train,
        "y_test": y_test,
        "scaler_1d": scaler_1d,
        "scaler_2d": scaler_2d,
        "file_paths": file_test  # Use test set paths for predicting demo
    }
