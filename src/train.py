import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras

# Set global reproducibility seed
def set_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

set_reproducibility(42)

from src.feature_extraction import EMOTION_MAP
from src.models import (
    build_cnn_model, 
    build_rnn_model, 
    build_lstm_model, 
    build_random_forest_model, 
    build_svm_model
)
from src.kathbath_pretrain import pretrain_on_kathbath, build_classifier_from_pretrained
from src.kathbath_dann import train_dann_model

def plot_training_history(history, model_name, output_dir="results/plots"):
    """
    Plots the training and validation loss and accuracy curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy curves
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='royalblue')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='darkorange')
    ax1.set_title(f'{model_name} Accuracy over Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Loss curves
    ax2.plot(history.history['loss'], label='Train Loss', color='royalblue')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Val Loss', color='darkorange')
    ax2.set_title(f'{model_name} Loss over Epochs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved learning curves to: {plot_path}")

def plot_confusion_matrix(y_true, y_pred, model_name, emotion_labels, output_dir="results/plots"):
    """
    Generates and saves a confusion matrix heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=emotion_labels, 
        yticklabels=emotion_labels
    )
    plt.title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix heatmap to: {cm_path}")

def run_training_pipeline(data_dict, epochs=30, batch_size=32, results_dir="results",
                          run_pretrain=False, run_dann=False,
                          kathbath_dir="/Users/apple/Downloads/kb_data_clean_m4a/telugu",
                          kathbath_limit=500):
    """
    Executes training and evaluation for all models.
    """
    models_dir = os.path.join(results_dir, "models")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data splits
    X_train_2d = data_dict["X_train_2d"]
    X_test_2d = data_dict["X_test_2d"]
    X_train_1d = data_dict["X_train_1d"]
    X_test_1d = data_dict["X_test_1d"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    
    # Save scalers first
    with open(os.path.join(models_dir, "scaler_1d.pkl"), "wb") as f:
        pickle.dump(data_dict["scaler_1d"], f)
    with open(os.path.join(models_dir, "scaler_2d.pkl"), "wb") as f:
        pickle.dump(data_dict["scaler_2d"], f)
        
    emotion_labels = [EMOTION_MAP[f"{i:02d}"][0] for i in range(1, 9)]
    
    # Performance summary tracker
    performance_summary = {}
    
    # --- MODEL 1: CNN ---
    print("\n" + "="*50 + "\n[INFO] Training CNN Model...\n" + "="*50)
    # Reshape X for 2D Conv: (samples, time_steps, features, channels)
    X_train_cnn = np.expand_dims(X_train_2d, axis=-1)
    X_test_cnn = np.expand_dims(X_test_2d, axis=-1)
    
    cnn_input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], 1)
    cnn_model = build_cnn_model(cnn_input_shape, num_classes=8)
    
    # Callback to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history_cnn = cnn_model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=min(epochs, 2),
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate CNN
    plot_training_history(history_cnn, "CNN Model", plots_dir)
    y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
    
    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    prec_cnn, rec_cnn, f1_cnn, _ = precision_recall_fscore_support(y_test, y_pred_cnn, average='weighted')
    plot_confusion_matrix(y_test, y_pred_cnn, "CNN Model", emotion_labels, plots_dir)
    
    performance_summary["CNN"] = {
        "Accuracy": acc_cnn,
        "Precision": prec_cnn,
        "Recall": rec_cnn,
        "F1-Score": f1_cnn,
        "object": cnn_model,
        "type": "dl_cnn"
    }
    
    # --- MODEL 2: RNN ---
    print("\n" + "="*50 + "\n[INFO] Training RNN Model...\n" + "="*50)
    rnn_input_shape = (X_train_2d.shape[1], X_train_2d.shape[2])
    rnn_model = build_rnn_model(rnn_input_shape, num_classes=8)
    
    history_rnn = rnn_model.fit(
        X_train_2d, y_train,
        validation_data=(X_test_2d, y_test),
        epochs=min(epochs, 2),
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate RNN
    plot_training_history(history_rnn, "RNN Model", plots_dir)
    y_pred_rnn_probs = rnn_model.predict(X_test_2d)
    y_pred_rnn = np.argmax(y_pred_rnn_probs, axis=1)
    
    acc_rnn = accuracy_score(y_test, y_pred_rnn)
    prec_rnn, rec_rnn, f1_rnn, _ = precision_recall_fscore_support(y_test, y_pred_rnn, average='weighted')
    plot_confusion_matrix(y_test, y_pred_rnn, "RNN Model", emotion_labels, plots_dir)
    
    performance_summary["RNN"] = {
        "Accuracy": acc_rnn,
        "Precision": prec_rnn,
        "Recall": rec_rnn,
        "F1-Score": f1_rnn,
        "object": rnn_model,
        "type": "dl_seq"
    }

    # --- MODEL 3: LSTM ---
    print("\n" + "="*50 + "\n[INFO] Training LSTM Model...\n" + "="*50)
    lstm_input_shape = (X_train_2d.shape[1], X_train_2d.shape[2])
    lstm_model = build_lstm_model(lstm_input_shape, num_classes=8)
    
    history_lstm = lstm_model.fit(
        X_train_2d, y_train,
        validation_data=(X_test_2d, y_test),
        epochs=min(epochs, 2),
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate LSTM
    plot_training_history(history_lstm, "LSTM Model", plots_dir)
    y_pred_lstm_probs = lstm_model.predict(X_test_2d)
    y_pred_lstm = np.argmax(y_pred_lstm_probs, axis=1)
    
    acc_lstm = accuracy_score(y_test, y_pred_lstm)
    prec_lstm, rec_lstm, f1_lstm, _ = precision_recall_fscore_support(y_test, y_pred_lstm, average='weighted')
    plot_confusion_matrix(y_test, y_pred_lstm, "LSTM Model", emotion_labels, plots_dir)
    
    performance_summary["LSTM"] = {
        "Accuracy": acc_lstm,
        "Precision": prec_lstm,
        "Recall": rec_lstm,
        "F1-Score": f1_lstm,
        "object": lstm_model,
        "type": "dl_seq"
    }
    
    # --- MODEL 4: Random Forest ---
    print("\n" + "="*50 + "\n[INFO] Training Random Forest Baseline...\n" + "="*50)
    rf_model = build_random_forest_model()
    rf_model.fit(X_train_1d, y_train)
    
    # Evaluate RF
    y_pred_rf = rf_model.predict(X_test_1d)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest", emotion_labels, plots_dir)
    
    performance_summary["Random Forest"] = {
        "Accuracy": acc_rf,
        "Precision": prec_rf,
        "Recall": rec_rf,
        "F1-Score": f1_rf,
        "object": rf_model,
        "type": "ml"
    }

    # --- MODEL 5: SVM ---
    print("\n" + "="*50 + "\n[INFO] Training & Tuning SVM Classifier...\n" + "="*50)
    from sklearn.model_selection import GridSearchCV
    raw_svm = build_svm_model()
    param_grid = {
        'C': [1, 10, 50, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001]
    }
    svm_cv = GridSearchCV(raw_svm, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    svm_cv.fit(X_train_1d, y_train)
    
    svm_model = svm_cv.best_estimator_
    print(f"[INFO] Best SVM hyperparameters selected: {svm_cv.best_params_}")
    
    # Evaluate SVM
    y_pred_svm = svm_model.predict(X_test_1d)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted')
    plot_confusion_matrix(y_test, y_pred_svm, "SVM Classifier", emotion_labels, plots_dir)
    
    performance_summary["SVM"] = {
        "Accuracy": acc_svm,
        "Precision": prec_svm,
        "Recall": rec_svm,
        "F1-Score": f1_svm,
        "object": svm_model,
        "type": "ml"
    }
    
    # --- MODEL 6: Pretrained CNN ---
    if run_pretrain:
        print("\n" + "="*50 + "\n[INFO] Running Pre-trained Autoencoder fine-tuning...\n" + "="*50)
        # We run autoencoder pre-training with fewer epochs (e.g. 5) to save CPU time
        encoder_path = pretrain_on_kathbath(data_dir=kathbath_dir, limit=kathbath_limit, epochs=5)
        pretrain_model = build_classifier_from_pretrained(encoder_path, num_classes=8)
        
        # Reshape X for 2D Conv
        X_train_cnn = np.expand_dims(X_train_2d, axis=-1)
        X_test_cnn = np.expand_dims(X_test_2d, axis=-1)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        history_pretrain = pretrain_model.fit(
            X_train_cnn, y_train,
            validation_data=(X_test_cnn, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        plot_training_history(history_pretrain, "Pre-trained CNN Model", plots_dir)
        y_pred_pretrain_probs = pretrain_model.predict(X_test_cnn)
        y_pred_pretrain = np.argmax(y_pred_pretrain_probs, axis=1)
        
        acc_pretrain = accuracy_score(y_test, y_pred_pretrain)
        prec_pretrain, rec_pretrain, f1_pretrain, _ = precision_recall_fscore_support(y_test, y_pred_pretrain, average='weighted')
        plot_confusion_matrix(y_test, y_pred_pretrain, "Pre-trained CNN", emotion_labels, plots_dir)
        
        performance_summary["Pre-trained CNN"] = {
            "Accuracy": acc_pretrain,
            "Precision": prec_pretrain,
            "Recall": rec_pretrain,
            "F1-Score": f1_pretrain,
            "object": pretrain_model,
            "type": "dl_cnn"
        }

    # --- MODEL 7: DANN ---
    if run_dann:
        dann_model, acc_dann = train_dann_model(
            X_train_2d, y_train, X_test_2d, y_test,
            kathbath_dir=kathbath_dir,
            kathbath_limit=kathbath_limit,
            epochs=epochs,
            batch_size=batch_size
        )
        
        X_test_cnn = np.expand_dims(X_test_2d, axis=-1)
        y_pred_dann_probs, _ = dann_model.predict(X_test_cnn)
        y_pred_dann = np.argmax(y_pred_dann_probs, axis=1)
        
        acc_dann = accuracy_score(y_test, y_pred_dann)
        prec_dann, rec_dann, f1_dann, _ = precision_recall_fscore_support(y_test, y_pred_dann, average='weighted')
        plot_confusion_matrix(y_test, y_pred_dann, "DANN Model", emotion_labels, plots_dir)
        
        performance_summary["DANN"] = {
            "Accuracy": acc_dann,
            "Precision": prec_dann,
            "Recall": rec_dann,
            "F1-Score": f1_dann,
            "object": dann_model,
            "type": "dl_dann"
        }
    
    # --- MODEL COMPARISON & SAVING BEST MODEL ---
    print("\n" + "="*50 + "\n[INFO] Comparing Model Performances...\n" + "="*50)
    
    comparison_data = []
    for model_name, metrics in performance_summary.items():
        comparison_data.append({
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"]
        })
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display the comparison table
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    
    # Plot comparison bar chart
    plt.figure(figsize=(10, 6))
    melted_df = pd.melt(comparison_df, id_vars="Model", var_name="Metric", value_name="Value")
    sns.barplot(x="Model", y="Value", hue="Metric", data=melted_df, palette="viridis")
    plt.title("Speech Emotion Recognition - Model Performance Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    comparison_plot_path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved comparison plot to: {comparison_plot_path}")
    
    # Select best model based on accuracy
    best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Model']
    best_model_data = performance_summary[best_model_name]
    best_model_obj = best_model_data["object"]
    
    print(f"\n[SUCCESS] Best Performing Model: {best_model_name} (Accuracy: {best_model_data['Accuracy']:.4f})")
    
    # Save the best model
    best_model_type = best_model_data["type"]
    best_model_meta = {
        "name": best_model_name,
        "type": best_model_type,
        "accuracy": best_model_data["Accuracy"]
    }
    
    # Save metadata
    with open(os.path.join(models_dir, "best_model_meta.pkl"), "wb") as f:
        pickle.dump(best_model_meta, f)
        
    if best_model_type.startswith("dl_"):
        model_path = os.path.join(models_dir, "best_model.keras")
        best_model_obj.save(model_path)
    else:
        model_path = os.path.join(models_dir, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model_obj, f)
            
    print(f"[INFO] Saved best model ({best_model_name}) to: {model_path}")
    
    # Generate overall classification reports for display
    print("\nDetailed Best Model Classification Report:")
    if best_model_type == "dl_cnn":
        y_pred = np.argmax(best_model_obj.predict(X_test_cnn), axis=1)
    elif best_model_type == "dl_dann":
        y_pred_probs, _ = best_model_obj.predict(X_test_cnn)
        y_pred = np.argmax(y_pred_probs, axis=1)
    elif best_model_type == "dl_seq":
        y_pred = np.argmax(best_model_obj.predict(X_test_2d), axis=1)
    else:
        y_pred = best_model_obj.predict(X_test_1d)
        
    report = classification_report(y_test, y_pred, target_names=emotion_labels)
    print(report)
    
    # Write report to file
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy: {best_model_data['Accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("======================\n")
        f.write(report)
    print(f"[INFO] Saved classification report to: {report_path}")
    
    return best_model_name, best_model_data["Accuracy"]

if __name__ == "__main__":
    # Dummy data test if run directly
    print("[INFO] Please run main.py to execute the full pipeline.")
