import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def build_cnn_model(input_shape, num_classes=8):
    """
    Builds a 2D Convolutional Neural Network (CNN) for emotion classification.
    Takes 2D feature sequences (time_steps, features) and adds a channel dimension.
    """
    model = keras.Sequential([
        # Input shape expected: (time_steps, feature_dim, 1)
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Fully Connected Block
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(num_classes, activation='softmax')
    ], name="CNN_Model")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_rnn_model(input_shape, num_classes=8):
    """
    Builds a Simple Recurrent Neural Network (RNN) for emotion classification.
    Takes 2D feature sequences (time_steps, features).
    """
    model = keras.Sequential([
        # Input shape expected: (time_steps, feature_dim)
        layers.Input(shape=input_shape),
        
        layers.SimpleRNN(64, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.SimpleRNN(64),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(num_classes, activation='softmax')
    ], name="RNN_Model")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_lstm_model(input_shape, num_classes=8):
    """
    Builds a Long Short-Term Memory (LSTM) model for emotion classification.
    Takes 2D feature sequences (time_steps, features).
    """
    model = keras.Sequential([
        # Input shape expected: (time_steps, feature_dim)
        layers.Input(shape=input_shape),
        
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.LSTM(64)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(num_classes, activation='softmax')
    ], name="LSTM_Model")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_random_forest_model(n_estimators=200, random_state=42):
    """
    Returns a Random Forest Classifier baseline.
    Takes 1D feature vectors.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state, 
        n_jobs=-1
    )

def build_svm_model(random_state=42):
    """
    Returns a Support Vector Machine Classifier baseline.
    Takes 1D feature vectors.
    """
    return SVC(
        C=10.0,
        kernel='rbf',
        probability=True,
        random_state=random_state
    )
