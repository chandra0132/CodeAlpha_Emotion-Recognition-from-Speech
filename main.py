import os
import argparse
import random
import glob
import numpy as np

# Suppress TensorFlow logging and warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
try:
    # Disable GPU and MPS (Metal) device visibility to prevent hangs in background tasks
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

# Set global reproducibility seed
def set_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        # Enable deterministic ops for TensorFlow 2.9+
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

set_reproducibility(42)

from src.download import download_and_extract
from src.eda import perform_eda
from src.feature_extraction import load_dataset_features
from src.train import run_training_pipeline
from src.predict import EmotionPredictor, print_prediction_results

def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition Project Orchestrator")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store dataset files")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store output models, plots and logs")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs for deep learning models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for deep learning models")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading/extraction if already done")
    parser.add_argument("--demo_file", type=str, help="Path to a custom wav file to test prediction")
    parser.add_argument("--pretrain", action="store_true", help="Perform Autoencoder pre-training on Kathbath dataset")
    parser.add_argument("--dann", action="store_true", help="Perform Domain Adversarial training for gender invariance")
    parser.add_argument("--kathbath_dir", type=str, default="/Users/apple/Downloads/kb_data_clean_m4a/telugu", help="Path to Kathbath dataset directory")
    parser.add_argument("--kathbath_limit", type=int, default=500, help="Max number of files to process from Kathbath dataset")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("      SPEECH EMOTION RECOGNITION - PIPELINE ORCHESTRATOR")
    print("="*80)

    # 1. Dataset Downloading & Unzipping
    if not args.skip_download:
        print("\n--- STEP 1: DOWNLOADING & EXTRACTING DATASET ---")
        success = download_and_extract(args.data_dir)
        if not success:
            print("[ERROR] Step 1 failed. Exiting pipeline.")
            return
    else:
        print("\n--- STEP 1: SKIPPING DOWNLOAD (User Option) ---")

    # 2. Exploratory Data Analysis (EDA)
    print("\n--- STEP 2: RUNNING EXPLORATORY DATA ANALYSIS (EDA) ---")
    plots_dir = os.path.join(args.results_dir, "plots")
    eda_success = perform_eda(args.data_dir, plots_dir)
    if not eda_success:
        print("[ERROR] Step 2 EDA failed. Exiting pipeline.")
        return

    # 3. Preprocessing & Feature Extraction
    print("\n--- STEP 3: PREPROCESSING & FEATURE EXTRACTION ---")
    try:
        data_dict = load_dataset_features(args.data_dir)
    except Exception as e:
        print(f"[ERROR] Step 3 Feature Extraction failed: {e}")
        return

    # 4. Model Training & Evaluation
    print("\n--- STEP 4: MODEL TRAINING & EVALUATION ---")
    try:
        best_name, best_acc = run_training_pipeline(
            data_dict, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            results_dir=args.results_dir,
            run_pretrain=args.pretrain,
            run_dann=args.dann,
            kathbath_dir=args.kathbath_dir,
            kathbath_limit=args.kathbath_limit
        )
    except Exception as e:
        print(f"[ERROR] Step 4 Model Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Prediction System Demo
    print("\n--- STEP 5: PREDICTION DEMO ---")
    demo_wav = args.demo_file
    
    # If no custom file is provided, pick a random audio clip from the dataset
    if not demo_wav:
        wav_files = glob.glob(os.path.join(args.data_dir, "Actor_*", "*.wav"))
        if wav_files:
            demo_wav = random.choice(wav_files)
            print(f"[INFO] No demo file provided. Randomly selected a file from dataset for testing:")
            print(f"       {demo_wav}")
        else:
            print("[WARNING] No audio files found to perform a prediction demo.")
            return

    try:
        predictor = EmotionPredictor(models_dir=os.path.join(args.results_dir, "models"))
        emotion, conf, probs = predictor.predict(demo_wav)
        
        # Display results
        print_prediction_results(emotion, conf, probs)
        
        # Print ground truth if it was selected from the dataset
        filename = os.path.basename(demo_wav)
        parts = filename.split('-')
        if len(parts) >= 3 and demo_wav in wav_files:
            from src.feature_extraction import EMOTION_MAP
            true_code = parts[2]
            if true_code in EMOTION_MAP:
                true_emotion = EMOTION_MAP[true_code][0]
                print(f"Ground Truth Emotion: {true_emotion.upper()}")
                if true_emotion.lower() == emotion.lower():
                    print("[SUCCESS] Model prediction matches Ground Truth!")
                else:
                    print("[INFO] Model prediction does not match Ground Truth (speech emotion classification is a subjective task).")
    except Exception as e:
        print(f"[ERROR] Demo prediction failed: {e}")

    print("\n" + "="*80)
    print("      PIPELINE EXECUTION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
