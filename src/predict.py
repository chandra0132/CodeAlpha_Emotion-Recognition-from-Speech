import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pickle
import argparse
import numpy as np
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
from tensorflow import keras

from src.feature_extraction import extract_audio_features, EMOTION_MAP

class EmotionPredictor:
    """
    Inference interface to load the trained speech emotion recognition model
    and predict emotions from raw audio files.
    """
    def __init__(self, models_dir="results/models"):
        self.models_dir = models_dir
        self.meta_path = os.path.join(models_dir, "best_model_meta.pkl")
        
        # Emotion labels in index order
        self.emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        
        # Check if SOTA XLS-R model exists
        self.xlsr_dir = os.path.join(models_dir, "xlsr-ser")
        if os.path.exists(self.xlsr_dir):
            self.model_type = "transformer"
            self.model_name = "XLS-R Speech Transformer (SOTA)"
            print(f"[INFO] Loading SOTA model: {self.model_name} from {self.xlsr_dir}")
            
            import torch
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
            
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.xlsr_dir)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.xlsr_dir)
            self.model.eval()
        else:
            if not os.path.exists(self.meta_path):
                raise FileNotFoundError(
                    f"Model metadata not found at {self.meta_path}. "
                    "Have you run the training script successfully?"
                )
                
            # Load best model metadata
            with open(self.meta_path, "rb") as f:
                self.meta = pickle.load(f)
                
            self.model_name = self.meta["name"]
            self.model_type = self.meta["type"]
            print(f"[INFO] Loading model: {self.model_name} (type: {self.model_type})")
            
            # Load scalers
            with open(os.path.join(models_dir, "scaler_1d.pkl"), "rb") as f:
                self.scaler_1d = pickle.load(f)
            with open(os.path.join(models_dir, "scaler_2d.pkl"), "rb") as f:
                self.scaler_2d = pickle.load(f)
                
            # Load model binary
            if self.model_type.startswith("dl_"):
                model_path = os.path.join(models_dir, "best_model.keras")
                self.model = keras.models.load_model(model_path)
            else:
                model_path = os.path.join(models_dir, "best_model.pkl")
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

    def predict(self, file_path):
        """
        Predicts the emotion of a given wav audio file.
        
        Returns:
            predicted_emotion: string (e.g. 'happy')
            confidence: float (0.0 to 1.0)
            probabilities: dict of {emotion_name: probability}
        """
        if self.model_type == "transformer":
            import torch
            import librosa
            
            # Load and resample audio to 16,000 Hz for XLS-R
            y, sr = librosa.load(file_path, sr=16000)
            
            # Process audio inputs
            inputs = self.feature_extractor(y, sampling_rate=16000, return_tensors="pt")
            
            # Predict
            with torch.no_grad():
                logits = self.model(**inputs).logits
                
            # Compute probabilities
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            
            pred_idx = np.argmax(probs)
            predicted_emotion = self.emotion_labels[pred_idx]
            confidence = probs[pred_idx]
            
            prob_dict = {self.emotion_labels[i]: float(probs[i]) for i in range(len(self.emotion_labels))}
            return predicted_emotion, confidence, prob_dict
            
        else:
            # Extract features (standard MFCC, Chroma, Mel, Contrast)
            f_2d, f_1d = extract_audio_features(file_path)
            if f_2d is None or f_1d is None:
                raise ValueError(f"Could not extract features from file: {file_path}")
                
            # Scale and shape features depending on model type
            if self.model_type == "dl_cnn" or self.model_type == "dl_dann":
                num_time, num_feats = f_2d.shape
                f_2d_flat = f_2d.reshape(-1, num_feats)
                f_2d_scaled_flat = self.scaler_2d.transform(f_2d_flat)
                f_2d_scaled = f_2d_scaled_flat.reshape(1, num_time, num_feats)
                model_input = np.expand_dims(f_2d_scaled, axis=-1)
                
                preds = self.model.predict(model_input, verbose=0)
                if self.model_type == "dl_dann":
                    probs = preds[0][0]
                else:
                    probs = preds[0]
                
            elif self.model_type == "dl_seq":
                num_time, num_feats = f_2d.shape
                f_2d_flat = f_2d.reshape(-1, num_feats)
                f_2d_scaled_flat = self.scaler_2d.transform(f_2d_flat)
                f_2d_scaled = f_2d_scaled_flat.reshape(1, num_time, num_feats)
                
                probs = self.model.predict(f_2d_scaled, verbose=0)[0]
                
            else: # traditional ML
                f_1d_scaled = self.scaler_1d.transform(f_1d.reshape(1, -1))
                probs = self.model.predict_proba(f_1d_scaled)[0]
                
            pred_idx = np.argmax(probs)
            predicted_emotion = self.emotion_labels[pred_idx]
            confidence = probs[pred_idx]
            
            prob_dict = {self.emotion_labels[i]: float(probs[i]) for i in range(len(self.emotion_labels))}
            return predicted_emotion, confidence, prob_dict

def print_prediction_results(predicted_emotion, confidence, prob_dict):
    """
    Utility to display prediction results with a text-based progress bar.
    """
    print("\n" + "="*50)
    print(f"Prediction Result for Audio File:")
    print("="*50)
    print(f"PREDICTED EMOTION : {predicted_emotion.upper()}")
    print(f"CONFIDENCE LEVEL  : {confidence*100:.2f}%")
    print("-"*50)
    print("Probability Distribution:")
    
    # Sort emotions by probability for display
    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    
    for emotion, prob in sorted_probs:
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{emotion.capitalize():<12} : {bar} {prob*100:5.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Emotion from Speech Audio File")
    parser.add_argument("--file", type=str, required=True, help="Path to input .wav audio file")
    parser.add_argument("--models_dir", type=str, default="results/models", help="Directory where trained models are saved")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"[ERROR] Input file not found: {args.file}")
    else:
        try:
            predictor = EmotionPredictor(models_dir=args.models_dir)
            emotion, conf, probs = predictor.predict(args.file)
            print_prediction_results(emotion, conf, probs)
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
