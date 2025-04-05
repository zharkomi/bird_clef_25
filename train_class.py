import os
from datetime import datetime

import numpy as np
import pandas as pd
import glob
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from collections import defaultdict
import matplotlib.pyplot as plt

from cuda import setup_cuda

setup_cuda()

# Configuration variables - edit these to match your environment
EMBEDDINGS_DIR = '/home/mikhail/prj/bird_clef_25/embeddings'
N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 30
RANDOM_STATE = 42

# Create timestamped directory for outputs
OUTPUT_DIR = os.path.join('train_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Function to collect all embedding files by species
def collect_embedding_files(base_dir):
    """
    Collect all embedding files from the specified directory structure.
    """
    species_files = {}

    # Iterate through each species directory
    for species_id in os.listdir(base_dir):
        species_dir = os.path.join(base_dir, species_id)
        if os.path.isdir(species_dir):
            # Get all pickle files in this directory
            embedding_files = glob.glob(os.path.join(species_dir, "*.pkl"))

            if embedding_files:  # Only add species if it has embedding files
                species_files[species_id] = embedding_files
                print(f"Found {len(embedding_files)} embedding files for species: {species_id}")

    return species_files


# Function to load embeddings from file
def load_embeddings(file_path):
    """
    Load pre-extracted embeddings from a pickle file
    Each file contains a list of embeddings, where each embedding is a 1D array of length 1024
    """
    try:
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)

        # Check if embeddings is a list
        if not isinstance(embeddings, list):
            print(f"WARNING: Expected a list, got {type(embeddings)} from {file_path}")
            return None

        # Filter out any invalid embeddings
        valid_embeddings = []
        for i, embedding in enumerate(embeddings):
            valid_embeddings.append(embedding['embedding'])

        if len(valid_embeddings) == 0:
            print(f"WARNING: No valid embeddings found in {file_path}")
            return None

        print(f"Loaded {len(valid_embeddings)} valid embeddings from {file_path}")
        return valid_embeddings

    except Exception as e:
        print(f"Error loading embeddings from {file_path}: {e}")
        return None


# Function to prepare dataset with stratified k-fold splitting
def prepare_kfold_datasets(species_files, n_splits=5, random_state=42):
    """
    Load all embeddings and prepare datasets for stratified k-fold cross-validation
    ensuring each species is evenly distributed across folds
    """
    all_embeddings = []
    all_labels = []
    all_stratify_labels = []  # For stratification

    # For each species, load all embeddings
    for species_idx, (species, files) in enumerate(species_files.items()):
        # Count embeddings for this species
        species_embeddings = []

        # Load all embeddings for this species
        print(f"Loading embeddings for species: {species}")
        for file_path in files:
            embeddings = load_embeddings(file_path)
            if embeddings is not None and len(embeddings) > 0:
                species_embeddings.extend(embeddings)

        # Check if we have embeddings for this species
        if not species_embeddings:
            print(f"WARNING: No valid embeddings for species '{species}'. Skipping.")
            continue

        # Add embeddings and labels
        num_embeddings = len(species_embeddings)
        all_embeddings.extend(species_embeddings)
        all_labels.extend([species] * num_embeddings)
        all_stratify_labels.extend([species_idx] * num_embeddings)  # Use species index for stratification

        print(f"  - Added {num_embeddings} embeddings for species {species}")

    # Check if we have any data left
    if not all_embeddings:
        raise ValueError("After loading embeddings, no data remains for training.")

    # Verify that all embeddings have the correct shape before creating the array
    embedding_length = 1024
    valid_embeddings = []
    valid_labels = []
    valid_stratify_labels = []

    print("Verifying embedding dimensions...")
    for i, (embedding, label, stratify_label) in enumerate(zip(all_embeddings, all_labels, all_stratify_labels)):
        if hasattr(embedding, 'shape') and embedding.shape == (embedding_length,):
            valid_embeddings.append(embedding)
            valid_labels.append(label)
            valid_stratify_labels.append(stratify_label)
        else:
            shape_info = embedding.shape if hasattr(embedding, 'shape') else type(embedding)
            print(f"WARNING: Removing embedding {i} with invalid shape: {shape_info}")

    if len(valid_embeddings) == 0:
        raise ValueError("No valid embeddings remain after validation.")

    # Convert to numpy arrays
    try:
        all_embeddings_array = np.stack(valid_embeddings)
        print(f"Successfully created embeddings array with shape: {all_embeddings_array.shape}")
    except Exception as e:
        print("Error creating embeddings array:", e)
        # Print debugging information
        for i, emb in enumerate(valid_embeddings[:5]):
            print(f"Embedding {i} type: {type(emb)}")
            if hasattr(emb, 'shape'):
                print(f"Embedding {i} shape: {emb.shape}")
            if hasattr(emb, 'dtype'):
                print(f"Embedding {i} dtype: {emb.dtype}")
        raise ValueError("Failed to create embeddings array. See above error.")

    all_stratify_labels = np.array(valid_stratify_labels)

    print(f"Total dataset: {len(valid_embeddings)} embeddings from {len(set(valid_labels))} species")
    print(f"Embedding shape: {all_embeddings_array.shape}")

    # Create stratified k-fold splitter
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return all_embeddings_array, valid_labels, kfold, all_stratify_labels


# Function to prepare embeddings and labels for a specific fold
def prepare_fold_data(embeddings, labels, train_indices, val_indices):
    """
    Prepare embeddings and labels for a specific fold from pre-loaded embeddings
    """
    # Check if indices are valid
    if len(train_indices) == 0:
        raise ValueError("No training indices provided")
    if len(val_indices) == 0:
        raise ValueError("No validation indices provided")

    # Split embeddings and labels
    train_embeddings = embeddings[train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_embeddings = embeddings[val_indices]
    val_labels = [labels[i] for i in val_indices]

    # Check if embeddings are properly formed
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Validation embeddings shape: {val_embeddings.shape}")

    # Create label encoder
    label_encoder = LabelEncoder()
    all_labels = train_labels + val_labels
    label_encoder.fit(all_labels)

    # Encode labels
    train_encoded_labels = label_encoder.transform(train_labels)
    val_encoded_labels = label_encoder.transform(val_labels)

    print(f"Prepared {len(train_embeddings)} training and {len(val_embeddings)} validation embeddings")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    return train_embeddings, train_encoded_labels, val_embeddings, val_encoded_labels, label_encoder


# Function to create TensorFlow dataset
def create_tf_dataset(embeddings, labels, batch_size=32, is_training=False):
    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))

    # Shuffle if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(embeddings))

    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


# Function to create TensorFlow model
def create_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Function to train the model
def train_model(train_dataset, val_dataset, input_dim, num_classes, fold_num, epochs=30):
    # Create model
    model = create_model(input_dim, num_classes)

    # Define callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        f'{OUTPUT_DIR}/best_species_classifier_fold{fold_num}.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    return model, history


# Function to evaluate the model
def evaluate_model(model, val_dataset, label_encoder):
    # Predict on validation dataset
    predictions = []
    true_labels = []

    for embeddings, labels in val_dataset:
        batch_predictions = model.predict(embeddings)
        predictions.extend(np.argmax(batch_predictions, axis=1))
        true_labels.extend(labels.numpy())

    # Convert numeric labels back to original species names
    pred_species = label_encoder.inverse_transform(predictions)
    true_species = label_encoder.inverse_transform(true_labels)

    # Calculate accuracy
    accuracy = accuracy_score(true_species, pred_species)

    # Create report with zero_division=0 parameter to fix the warning
    report = classification_report(true_species, pred_species, output_dict=True, zero_division=0)

    return accuracy, report, pred_species, true_species


# Function to plot training history
def plot_history(history, fold_num):
    """Plot training and validation metrics for a fold"""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - Fold {fold_num}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - Fold {fold_num}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_history_fold{fold_num}.png')
    plt.close()


def main():
    print("Starting bird species classification training with stratified k-fold cross-validation")
    print(f"Configuration: {N_SPLITS} folds, {BATCH_SIZE} batch size, {EPOCHS} max epochs")

    # Step 1: Collect embedding files for each species
    print("Collecting embedding files...")
    species_files = collect_embedding_files(EMBEDDINGS_DIR)

    # Print summary of collected files
    print(f"Found {len(species_files)} species with embedding files:")
    for species, files in species_files.items():
        print(f"  - {species}: {len(files)} files")

    # Step 2: Load all embeddings and prepare for stratified k-fold cross-validation
    print(f"\nLoading all embeddings and preparing {N_SPLITS}-fold stratified cross-validation...")
    all_embeddings, all_labels, kfold, stratify_labels = prepare_kfold_datasets(
        species_files, n_splits=N_SPLITS, random_state=RANDOM_STATE
    )

    # Track metrics across folds
    fold_accuracies = []
    fold_histories = []
    all_reports = []

    # Step 3: Perform stratified k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_embeddings, stratify_labels)):
        print(f"\n===== Fold {fold + 1}/{N_SPLITS} =====")

        # Prepare data for this fold
        train_embeddings, train_labels, val_embeddings, val_labels, label_encoder = prepare_fold_data(
            all_embeddings, all_labels, train_idx, val_idx
        )

        # Check if we have enough data to continue
        if len(train_embeddings) == 0 or len(val_embeddings) == 0:
            print(f"Skipping fold {fold + 1} due to insufficient valid embeddings.")
            continue

        # Get input dimension and number of classes
        if len(train_embeddings.shape) < 2:
            print(f"ERROR: Train embeddings has invalid shape: {train_embeddings.shape}")
            print("Sample of train embeddings:")
            for i in range(min(5, len(train_embeddings))):
                print(f"  Embedding {i}: {train_embeddings[i]}")
            continue

        input_dim = train_embeddings.shape[1]
        num_classes = len(label_encoder.classes_)

        # Create TensorFlow datasets
        train_dataset = create_tf_dataset(train_embeddings, train_labels, BATCH_SIZE, is_training=True)
        val_dataset = create_tf_dataset(val_embeddings, val_labels, BATCH_SIZE)

        # Train model
        print(f"\nTraining model for fold {fold + 1}...")
        model, history = train_model(train_dataset, val_dataset, input_dim, num_classes, fold + 1, epochs=EPOCHS)

        # Evaluate model
        print(f"\nEvaluating model for fold {fold + 1}...")
        accuracy, report, pred_species, true_species = evaluate_model(model, val_dataset, label_encoder)

        # Print summary for this fold
        print(f"\nFold {fold + 1} Results:")
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Print top 5 misclassified species (if any misclassifications exist)
        misclassified = pd.DataFrame({
            'true': true_species,
            'predicted': pred_species
        })
        misclassified = misclassified[misclassified['true'] != misclassified['predicted']]
        if len(misclassified) > 0:
            print("\nTop misclassified species:")
            top_errors = misclassified.groupby(['true', 'predicted']).size().reset_index(name='count')
            top_errors = top_errors.sort_values('count', ascending=False).head(5)
            print(top_errors)

        # Store results
        fold_accuracies.append(accuracy)
        fold_histories.append(history)
        all_reports.append(report)

        # Plot training history
        plot_history(history, fold + 1)

        # Save model and label encoder for this fold
        model.save(f'{OUTPUT_DIR}/species_classifier_fold{fold + 1}.keras')
        with open(f'{OUTPUT_DIR}/species_label_encoder_fold{fold + 1}.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

    # Step 4: Summarize results across all folds
    print("\n===== Stratified K-Fold Cross-Validation Summary =====")
    print(f"Number of folds: {N_SPLITS}")
    print(f"Average Validation Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")

    # Calculate and print averaged metrics
    avg_metrics = defaultdict(list)

    # Collect metrics from all reports
    for report in all_reports:
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):  # Skip 'accuracy', 'macro avg', 'weighted avg'
                for metric_name, value in metrics.items():
                    avg_metrics[f"{class_name}_{metric_name}"].append(value)

    # Calculate and print overall metrics
    print("\nOverall Classification Metrics:")
    overall_metrics = {'precision': [], 'recall': [], 'f1-score': []}

    for report in all_reports:
        if 'weighted avg' in report:
            for metric in overall_metrics:
                if metric in report['weighted avg']:
                    overall_metrics[metric].append(report['weighted avg'][metric])

    for metric, values in overall_metrics.items():
        if values:
            print(f"  Weighted {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Save best model based on validation accuracy
    best_fold = np.argmax(fold_accuracies) + 1
    print(f"\nBest performing model was from fold {best_fold} with accuracy {fold_accuracies[best_fold - 1]:.4f}")

    # Copy best model to final model file
    import shutil
    shutil.copy(f'{OUTPUT_DIR}/species_classifier_fold{best_fold}.keras', f'{OUTPUT_DIR}/species_classifier_final.keras')
    shutil.copy(f'{OUTPUT_DIR}/species_label_encoder_fold{best_fold}.pkl', f'{OUTPUT_DIR}/species_label_encoder_final.pkl')

    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()