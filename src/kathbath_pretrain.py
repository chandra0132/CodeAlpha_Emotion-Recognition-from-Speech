import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.kathbath_loader import load_kathbath_dataset

def build_conv1d_autoencoder(input_shape):
    """
    Builds a 1D-CNN Autoencoder for sequence reconstruction.
    Input Shape: (time_steps, features) -> (130, 187)
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- ENCODER ---
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x) # Shape: (65, 64)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2)(x) # Shape: (32, 32)
    
    # --- DECODER ---
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x) # Shape: (64, 32)
    x = layers.ZeroPadding1D(padding=(0, 1))(x) # Shape: (65, 32) to match size
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x) # Shape: (130, 64)
    decoded = layers.Conv1D(input_shape[1], 3, activation='linear', padding='same')(x) # Shape: (130, 187)
    
    autoencoder = keras.Model(inputs, decoded, name="Autoencoder")
    encoder = keras.Model(inputs, encoded, name="Encoder")
    return autoencoder, encoder

def pretrain_on_kathbath(data_dir="/Users/apple/Downloads/kb_data_clean_m4a/telugu", limit=500, epochs=15):
    """
    Runs self-supervised pre-training on Kathbath dataset.
    """
    print("\n" + "="*50 + "\n[INFO] Starting Autoencoder Pre-training on Kathbath...\n" + "="*50)
    
    # Load Kathbath features (speaker/gender labels are ignored here)
    X_kathbath, _, _ = load_kathbath_dataset(data_dir=data_dir, limit=limit)
    
    if len(X_kathbath) == 0:
        raise ValueError(f"No valid features extracted from Kathbath dataset at: {data_dir}")
        
    input_shape = (X_kathbath.shape[1], X_kathbath.shape[2]) # (130, 187)
    
    autoencoder, encoder = build_conv1d_autoencoder(input_shape)
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Print architecture
    autoencoder.summary()
    
    # Standardize data locally for reconstruction
    # We compute mean/std along the whole dataset for normalization
    mean = np.mean(X_kathbath, axis=(0, 1), keepdims=True)
    std = np.std(X_kathbath, axis=(0, 1), keepdims=True) + 1e-8
    X_kathbath_norm = (X_kathbath - mean) / std
    
    # Fit Autoencoder
    autoencoder.fit(
        X_kathbath_norm, X_kathbath_norm,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Save the encoder model & normalization parameters
    models_dir = "results/models"
    os.makedirs(models_dir, exist_ok=True)
    
    encoder_path = os.path.join(models_dir, "kathbath_encoder.keras")
    encoder.save(encoder_path)
    
    # Save normalization statistics
    np.save(os.path.join(models_dir, "kathbath_norm_stats.npy"), {"mean": mean, "std": std})
    
    print(f"[SUCCESS] Pre-trained encoder saved to: {encoder_path}")
    return encoder_path

def build_classifier_from_pretrained(encoder_path, num_classes=8):
    """
    Loads pre-trained encoder and builds a classification model on top.
    """
    print(f"[INFO] Initializing classifier with pre-trained encoder from: {encoder_path}")
    encoder = keras.models.load_model(encoder_path)
    
    # We keep encoder trainable but could also freeze it (encoder.trainable = False)
    encoder.trainable = True
    
    inputs = encoder.input
    features = encoder.output # Shape: (32, 32)
    
    # Classification head
    x = layers.Flatten()(features)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    classifier = keras.Model(inputs=inputs, outputs=outputs, name="Pretrained_Classifier")
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005), # Slightly lower learning rate for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return classifier
