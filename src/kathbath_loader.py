import os
import numpy as np
from src.feature_extraction import extract_audio_features

def load_kathbath_dataset(data_dir="/Users/apple/Downloads/kb_data_clean_m4a/telugu", limit=500):
    """
    Parses the Kathbath directory structure recursively, extracts features for all audio clips,
    and returns features along with speaker and gender labels.
    """
    print(f"[INFO] Scanning directory: {data_dir} for Kathbath files...")
    features_list_2d = []
    genders = []      # 0 for male, 1 for female
    speaker_ids = []
    
    # Traverse directories recursively to find m4a audio files
    m4a_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".m4a"):
                m4a_files.append(os.path.join(root, file))
                
    print(f"[INFO] Found {len(m4a_files)} total .m4a files.")
    
    if limit and limit < len(m4a_files):
        m4a_files = m4a_files[:limit]
        print(f"[INFO] Capping extraction at user limit: {limit} files.")
        
    count = 0
    for file_path in m4a_files:
        filename = os.path.basename(file_path)
        # Expected naming: <speaker_id>-<utterance_id>-<gender>.m4a
        parts = filename.replace('.m4a', '').split('-')
        
        if len(parts) >= 3:
            speaker_id = parts[0]
            gender_char = parts[2].lower() # 'm' or 'f'
            
            # Extract features using existing feature extractor
            f_2d, _ = extract_audio_features(file_path)
            
            if f_2d is not None:
                features_list_2d.append(f_2d)
                genders.append(0 if gender_char == 'm' else 1)
                speaker_ids.append(speaker_id)
                
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(m4a_files)} files...")
            
    print(f"[INFO] Feature extraction completed. Successfully processed {len(features_list_2d)} files.")
    return np.array(features_list_2d), np.array(genders), speaker_ids
