import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.kathbath_loader import load_kathbath_dataset

# Custom Gradient Reversal Layer (GRL) in TensorFlow
@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        # Reverse the gradient direction by multiplying by -1.0
        return -dy
    return y, custom_grad

class GradReverseLayer(layers.Layer):
    def call(self, x):
        return grad_reverse(x)

def build_dann_model(input_shape, num_emotions=8):
    """
    Builds the DANN model structure:
    - Shared feature extractor: Extracts speech descriptors.
    - Emotion classification head: Softmax output for 8 emotion classes.
    - Gender classification head: Softmax output for 2 gender classes (passed through GRL).
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. Shared Feature Extractor
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    feature_rep = layers.Dense(128, activation='relu')(x)
    feature_rep = layers.BatchNormalization()(feature_rep)
    
    # 2. Emotion Classifier Branch
    emotion_branch = layers.Dense(64, activation='relu')(feature_rep)
    emotion_branch = layers.BatchNormalization()(emotion_branch)
    emotion_branch = layers.Dropout(0.4)(emotion_branch)
    emotion_out = layers.Dense(num_emotions, activation='softmax', name='emotion_output')(emotion_branch)
    
    # 3. Gender/Domain Classifier Branch (Passes through GRL)
    grl_branch = GradReverseLayer()(feature_rep)
    gender_branch = layers.Dense(64, activation='relu')(grl_branch)
    gender_branch = layers.BatchNormalization()(gender_branch)
    gender_branch = layers.Dropout(0.4)(gender_branch)
    gender_out = layers.Dense(2, activation='softmax', name='gender_output')(gender_branch)
    
    model = keras.Model(inputs=inputs, outputs=[emotion_out, gender_out], name="DANN_Model")
    return model

def train_dann_model(X_emotion_train, y_emotion_train, X_emotion_test, y_emotion_test,
                     kathbath_dir="/Users/apple/Downloads/kb_data_clean_m4a/telugu",
                     kathbath_limit=500, epochs=15, batch_size=32):
    """
    Joint custom training loop to perform Domain Adversarial training:
    - Minimizes loss for RAVDESS emotion predictions.
    - Maximizes loss (reverses gradients) for Kathbath gender predictions.
    """
    print("\n" + "="*50 + "\n[INFO] Starting Domain Adversarial (DANN) Training...\n" + "="*50)
    
    # 1. Load Kathbath features for gender bias learning
    X_gender, y_gender, _ = load_kathbath_dataset(data_dir=kathbath_dir, limit=kathbath_limit)
    
    if len(X_gender) == 0:
        raise ValueError(f"No valid features found in Kathbath dataset path: {kathbath_dir}")
        
    # Standardize Kathbath data locally to match RAVDESS features range
    mean = np.mean(X_gender, axis=(0, 1), keepdims=True)
    std = np.std(X_gender, axis=(0, 1), keepdims=True) + 1e-8
    X_gender_norm = (X_gender - mean) / std
    
    input_shape = (X_emotion_train.shape[1], X_emotion_train.shape[2], 1)
    model = build_dann_model(input_shape, num_emotions=8)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    loss_emotion_fn = keras.losses.SparseCategoricalCrossentropy()
    loss_gender_fn = keras.losses.SparseCategoricalCrossentropy()
    
    # Track statistics
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    
    num_samples = min(len(X_emotion_train), len(X_gender_norm))
    current_batch_size = min(batch_size, num_samples)
    steps_per_epoch = max(1, num_samples // current_batch_size)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Shuffle index orders for both datasets
        indices_e = np.random.permutation(len(X_emotion_train))
        indices_g = np.random.permutation(len(X_gender_norm))
        
        epoch_loss_emotion = 0.0
        epoch_loss_gender = 0.0
        train_acc_metric.reset_state()
        
        for step in range(steps_per_epoch):
            # Batch extraction
            batch_e_idx = indices_e[step*current_batch_size : (step+1)*current_batch_size]
            batch_g_idx = indices_g[step*current_batch_size : (step+1)*current_batch_size]
            
            x_e_batch = X_emotion_train[batch_e_idx] # Shape: (batch_size, 130, 187)
            y_e_batch = y_emotion_train[batch_e_idx]
            
            x_g_batch = X_gender_norm[batch_g_idx] # Shape: (batch_size, 130, 187)
            y_g_batch = y_gender[batch_g_idx]
            
            # Reshape to 4D Conv shape: (batch_size, 130, 187, 1)
            x_e_input = np.expand_dims(x_e_batch, axis=-1)
            x_g_input = np.expand_dims(x_g_batch, axis=-1)
            
            # Gradient tape tracking
            with tf.GradientTape() as tape:
                # Emotion prediction loss (RAVDESS batch)
                pred_emotion, _ = model(x_e_input, training=True)
                loss_emotion = loss_emotion_fn(y_e_batch, pred_emotion)
                
                # Gender prediction loss (Kathbath batch)
                _, pred_gender = model(x_g_input, training=True)
                loss_gender = loss_gender_fn(y_g_batch, pred_gender)
                
                # Combined loss: optimizer tries to minimize both
                # But GRL reverses gradients for loss_gender, making features gender-invariant
                total_loss = loss_emotion + 0.5 * loss_gender
                
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Update metric and losses
            train_acc_metric.update_state(y_e_batch, pred_emotion)
            epoch_loss_emotion += float(loss_emotion)
            epoch_loss_gender += float(loss_gender)
            
        # Validation evaluation on RAVDESS testing set
        X_test_input = np.expand_dims(X_emotion_test, axis=-1)
        val_pred_emotion, _ = model(X_test_input, training=False)
        val_acc = np.mean(np.argmax(val_pred_emotion, axis=1) == y_emotion_test)
        
        print(f"  Train Acc: {train_acc_metric.result():.4f} | Val Acc: {val_acc:.4f} | Emotion Loss: {epoch_loss_emotion/steps_per_epoch:.4f} | Gender Loss: {epoch_loss_gender/steps_per_epoch:.4f}")
        
    # Save the DANN model
    models_dir = "results/models"
    os.makedirs(models_dir, exist_ok=True)
    dann_path = os.path.join(models_dir, "best_model.keras" if val_acc > 0.5 else "dann_model.keras")
    model.save(dann_path)
    print(f"[SUCCESS] DANN model saved successfully to: {dann_path}")
    
    return model, val_acc
