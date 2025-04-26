import os
import pickle
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, callbacks
from sklearn.metrics import classification_report, accuracy_score

from cuda import setup_cuda
from src.utils import custom_loss
from train_prep import collect_embedding_files, prepare_kfold_datasets, prepare_fold_data, create_tf_dataset, N_SPLITS, \
    prepare_data, check_fold_status

setup_cuda()

# Configuration variables - edit these to match your environment
BATCH_SIZE = 32
EPOCHS = 30
USE_ENSEMBLE = True  # Enable/disable ensemble model creation

# Create timestamped directory for outputs
OUTPUT_DIR = "train"


def one_cycle_lr(epoch, max_lr=0.005, min_lr=0.0001, total_epochs=30):
    """One-cycle learning rate schedule for better convergence."""
    if epoch < total_epochs // 2:
        return min_lr + (max_lr - min_lr) * (epoch / (total_epochs // 2))
    else:
        return max_lr - (max_lr - min_lr) * ((epoch - total_epochs // 2) / (total_epochs // 2))


def create_model(input_dim=1024, num_classes=206, dropout_rate=0.4):
    """Create an optimized bird species classification model with residual connections"""
    inputs = layers.Input(shape=(input_dim,), name='input_embeddings')

    # First block with increased regularization
    x = layers.Dense(384, kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)  # Higher dropout in first layer (0.5)

    # Store for residual connection
    x_prev = x

    # Second block with residual connection
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Add residual connection with projection
    x_res = layers.Dense(256)(x_prev)  # Project to match dimensions
    x = layers.Add()([x, x_res])

    # Third block with residual connection
    x_prev = x
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Add another residual connection with projection
    x_res = layers.Dense(128)(x_prev)  # Project to match dimensions
    x = layers.Add()([x, x_res])

    # Output layer
    outputs = layers.Dense(num_classes, name='species_logits')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Use gradient clipping with Adam optimizer and learning rate schedule
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0  # Add gradient clipping to prevent exploding gradients
    )

    # Compile with the custom loss function from src.utils
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    return model


# Function to train the model with enhanced callbacks
def train_model(train_dataset, val_dataset, input_dim, num_classes, fold_num, epochs=30):
    # Create optimized model
    model = create_model(input_dim, num_classes)

    # Define enhanced callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        f'{OUTPUT_DIR}/best_species_classifier_fold{fold_num}.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    # Enhanced early stopping with more patience
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,  # Increased from 5 to 7
        restore_best_weights=True,
        verbose=1
    )

    # Add learning rate scheduler using one-cycle policy
    lr_scheduler = callbacks.LearningRateScheduler(one_cycle_lr)

    # Add ReduceLROnPlateau as a backup strategy
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Train model with enhanced callbacks
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler, reduce_lr]
    )

    return model, history


# BirdNET-style sigmoid function
def flat_sigmoid(x, sensitivity=-1.0):
    """Apply the same custom sigmoid function used by BirdNET"""
    return 1.0 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))


# Function to evaluate the model
def evaluate_model(model, val_dataset, label_encoder):
    # Predict on validation dataset
    predictions = []
    true_labels = []

    for embeddings, labels in val_dataset:
        # Get raw logits from the model
        batch_logits = model.predict(embeddings)
        # Apply sigmoid transformation similar to BirdNET
        batch_confidences = flat_sigmoid(batch_logits, sensitivity=-1.0)
        # Get predicted class indices from the confidences
        batch_predictions = np.argmax(batch_confidences, axis=1)

        predictions.extend(batch_predictions)
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


# EnsembleModel class
class EnsembleModel(tf.keras.Model):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def call(self, inputs):
        # Get predictions from each model
        predictions = [model(inputs) for model in self.models]
        # Stack and average predictions
        stacked = tf.stack(predictions, axis=0)
        return tf.reduce_mean(stacked, axis=0)


# Function to create an ensemble model from all fold models
def create_ensemble_model(fold_models, input_shape, output_dir):
    """
    Create an ensemble model by averaging predictions from multiple fold models.
    Saves the model in a deployment-friendly format.
    """
    # Create ensemble model
    ensemble_model = EnsembleModel(fold_models)

    # Compile the model to ensure it has optimizer, loss, and metrics
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Build the model with the correct input shape
    dummy_input = tf.random.normal((1, input_shape))
    _ = ensemble_model(dummy_input)  # Force build

    # Create a concrete function with the correct input signature for saving
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_shape], dtype=tf.float32)])
    def ensemble_predict(inputs):
        return ensemble_model(inputs)

    # Export as a SavedModel (standard deployment format)
    tf.saved_model.save(
        ensemble_model,
        f'{output_dir}/species_classifier_ensemble',
        signatures={'serving_default': ensemble_predict}
    )

    # Create TFLite version (for mobile/edge deployment)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [ensemble_predict.get_concrete_function()], ensemble_model
    )
    tflite_model = converter.convert()

    # Save TFLite model
    with open(f'{output_dir}/species_classifier_ensemble.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"Ensemble model saved to {output_dir}/species_classifier_ensemble")
    print(f"TFLite version saved to {output_dir}/species_classifier_ensemble.tflite")

    return ensemble_model


# Function to evaluate the ensemble model
def evaluate_ensemble_model(ensemble_model, val_datasets, label_encoders, output_dir):
    """
    Evaluate the ensemble model on combined validation data from all folds.
    """
    print("\n===== Evaluating Ensemble Model =====")

    # First, we need to handle potential differences in label encoders across folds
    # We'll use the label encoder from the last fold as our reference
    reference_label_encoder = label_encoders[-1]

    # Create a mapping between different fold label indices if needed
    mappings = []
    for le in label_encoders:
        if set(le.classes_) != set(reference_label_encoder.classes_):
            print("WARNING: Label encoders differ between folds. Using mapping.")
            mapping = {}
            for i, cls in enumerate(le.classes_):
                mapping[i] = reference_label_encoder.transform([cls])[0]
            mappings.append(mapping)
        else:
            mappings.append(None)

    # Save the reference label encoder for the ensemble
    with open(f'{output_dir}/species_label_encoder_ensemble.pkl', 'wb') as f:
        pickle.dump(reference_label_encoder, f)

    # Collect all validation predictions and true labels
    all_predictions = []
    all_true_labels = []

    # Process each fold's validation data
    for fold_idx, (val_dataset, label_encoder, mapping) in enumerate(zip(val_datasets, label_encoders, mappings)):
        fold_predictions = []
        fold_true_labels = []

        # Make predictions on this fold's validation data
        for embeddings, labels in val_dataset:
            # Get ensemble predictions (raw logits)
            batch_logits = ensemble_model(embeddings)
            # Apply sigmoid transformation similar to BirdNET
            batch_confidences = flat_sigmoid(batch_logits, sensitivity=-1.0)
            # Get predicted class indices
            predicted_indices = np.argmax(batch_confidences, axis=1)

            # Map indices if needed
            if mapping:
                predicted_indices = np.array([mapping.get(idx, idx) for idx in predicted_indices])
                fold_true_mapped = np.array([mapping.get(idx, idx) for idx in labels.numpy()])
                fold_true_labels.extend(fold_true_mapped)
            else:
                fold_true_labels.extend(labels.numpy())

            fold_predictions.extend(predicted_indices)

        # Add this fold's results to the overall results
        all_predictions.extend(fold_predictions)
        all_true_labels.extend(fold_true_labels)

    # Convert numeric predictions to species names
    pred_species = reference_label_encoder.inverse_transform(all_predictions)
    true_species = reference_label_encoder.inverse_transform(all_true_labels)

    # Calculate accuracy and generate report
    accuracy = accuracy_score(true_species, pred_species)
    report = classification_report(true_species, pred_species, output_dict=True, zero_division=0)

    print(f"Ensemble Model Validation Accuracy: {accuracy:.4f}")

    # Print top 5 misclassified species
    misclassified = pd.DataFrame({
        'true': true_species,
        'predicted': pred_species
    })
    misclassified = misclassified[misclassified['true'] != misclassified['predicted']]
    if len(misclassified) > 0:
        print("\nTop misclassified species (Ensemble Model):")
        top_errors = misclassified.groupby(['true', 'predicted']).size().reset_index(name='count')
        top_errors = top_errors.sort_values('count', ascending=False).head(5)
        print(top_errors)

    return accuracy, report


def train_fold(fold_idx, fold_data, batch_size, epochs, output_dir):
    """
    Train a model for a specific fold.

    Args:
        fold_idx: Index of the fold
        fold_data: Tuple containing training and validation data
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        output_dir: Directory to save outputs

    Returns:
        tuple: (model, history, accuracy, report, val_dataset, label_encoder) results of training
    """
    print(f"\n===== Fold {fold_idx + 1}/{N_SPLITS} =====")

    train_embeddings, train_labels, val_embeddings, val_labels, label_encoder = fold_data

    # Check if we have enough data to continue
    if len(train_embeddings) == 0 or len(val_embeddings) == 0:
        print(f"Skipping fold {fold_idx + 1} due to insufficient valid embeddings.")
        return None, None, None, None, None, None

    # Get input dimension and number of classes
    if len(train_embeddings.shape) < 2:
        print(f"ERROR: Train embeddings has invalid shape: {train_embeddings.shape}")
        print("Sample of train embeddings:")
        for i in range(min(5, len(train_embeddings))):
            print(f"  Embedding {i}: {train_embeddings[i]}")
        return None, None, None, None, None, None

    input_dim = train_embeddings.shape[1]
    num_classes = len(label_encoder.classes_)

    # Create TensorFlow datasets using the function from train_prep
    train_dataset = create_tf_dataset(train_embeddings, train_labels, batch_size, is_training=True)
    val_dataset = create_tf_dataset(val_embeddings, val_labels, batch_size)

    # Train model
    print(f"\nTraining model for fold {fold_idx + 1}...")
    model, history = train_model(train_dataset, val_dataset, input_dim, num_classes, fold_idx + 1, epochs=epochs)

    # Evaluate model
    print(f"\nEvaluating model for fold {fold_idx + 1}...")
    accuracy, report, pred_species, true_species = evaluate_model(model, val_dataset, label_encoder)

    # Print summary for this fold
    print(f"\nFold {fold_idx + 1} Results:")
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

    # Plot training history
    plot_history(history, fold_idx + 1)

    # Save model and label encoder for this fold in deployment-friendly format
    # Save using the model.save method to include Keras metadata
    model.save(f'{output_dir}/species_classifier_fold{fold_idx + 1}_keras')

    # Also save in SavedModel format if needed for other purposes
    tf.saved_model.save(model, f'{output_dir}/species_classifier_fold{fold_idx + 1}')

    # Save label encoder
    with open(f'{output_dir}/species_label_encoder_fold{fold_idx + 1}.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return model, history, accuracy, report, val_dataset, label_encoder


def load_trained_fold(fold_idx, train_dir, batch_size):
    """
    Load a previously trained model and its associated data without evaluation.

    Args:
        fold_idx: Index of the fold to load
        train_dir: Directory where fold data is stored
        batch_size: Batch size for validation dataset

    Returns:
        tuple: (model, val_dataset, label_encoder) loaded data
    """
    try:
        # Load the model
        model_path = os.path.join(train_dir, f"species_classifier_fold{fold_idx + 1}_keras")
        print(f"Loading model for fold {fold_idx + 1} from {model_path}...")
        custom_objects = {'custom_loss': custom_loss}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        # Load label encoder
        le_path = os.path.join(train_dir, f"species_label_encoder_fold{fold_idx + 1}.pkl")
        with open(le_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # Load fold data to recreate validation dataset
        fold_file = os.path.join(train_dir, f"fold_{fold_idx + 1}.pkl")
        with open(fold_file, "rb") as f:
            fold_data = pickle.load(f)

        # Recreate validation dataset using function from train_prep
        _, _, val_embeddings, val_labels, _ = fold_data
        val_dataset = create_tf_dataset(val_embeddings, val_labels, batch_size)

        print(f"Successfully loaded fold {fold_idx + 1} model")
        return model, val_dataset, label_encoder

    except Exception as e:
        print(f"Error loading model for fold {fold_idx + 1}: {e}")
        return None, None, None


def summarize_results(fold_accuracies, all_reports, output_dir):
    """
    Summarize results across all folds and save the best model.

    Args:
        fold_accuracies: List of validation accuracies for each fold
        all_reports: List of classification reports for each fold
        output_dir: Directory to save outputs
    """
    print("\n===== Stratified K-Fold Cross-Validation Summary =====")
    print(f"Number of folds: {len(fold_accuracies)}")

    if not fold_accuracies:
        print("No valid fold results to summarize.")
        return

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
    import shutil
    best_fold = np.argmax(fold_accuracies) + 1
    print(f"\nBest performing model was from fold {best_fold} with accuracy {fold_accuracies[best_fold - 1]:.4f}")

    # Copy the best Keras model file
    if os.path.exists(f'{output_dir}/species_classifier_fold{best_fold}_keras'):
        # Copy the Keras model directory
        if os.path.exists(f'{output_dir}/species_classifier_final'):
            shutil.rmtree(f'{output_dir}/species_classifier_final')
        shutil.copytree(f'{output_dir}/species_classifier_fold{best_fold}_keras',
                        f'{output_dir}/species_classifier_final')
        print(f"Copied Keras model from fold {best_fold} to final")

    # Copy label encoder
    shutil.copy(f'{output_dir}/species_label_encoder_fold{best_fold}.pkl',
                f'{output_dir}/species_label_encoder_final.pkl')
    print(f"Copied label encoder from fold {best_fold} to final")


def create_final_tflite_model(output_dir):
    """
    Create TFLite version of the best individual model.

    Args:
        output_dir: Directory where the best model is saved
    """
    print("\nCreating TFLite version of the best individual model...")

    # Use tf.keras.models.load_model instead of tf.saved_model.load
    custom_objects = {'custom_loss': custom_loss}
    best_model = tf.keras.models.load_model(
        f'{output_dir}/species_classifier_final',
        custom_objects=custom_objects
    )

    # Convert to TFLite (with optimization for size and latency)
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save TFLite model
    with open(f'{output_dir}/species_classifier_final.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite version of best model saved to {output_dir}/species_classifier_final.tflite")


def create_and_evaluate_ensemble(fold_models, val_datasets, label_encoders, input_dim, output_dir, best_fold_acc):
    """
    Create and evaluate an ensemble model from multiple fold models.

    Args:
        fold_models: List of trained models from each fold
        val_datasets: List of validation datasets for each fold
        label_encoders: List of label encoders for each fold
        input_dim: Input dimension for the model
        output_dir: Directory to save outputs
        best_fold_acc: Accuracy of the best individual fold model
    """
    print("\n===== Creating and Evaluating Ensemble Model =====")

    # Create ensemble model
    ensemble_model = create_ensemble_model(fold_models, input_dim, output_dir)

    # Evaluate ensemble model
    ensemble_acc, ensemble_report = evaluate_ensemble_model(
        ensemble_model,
        val_datasets,
        label_encoders,
        output_dir
    )

    # Compare with best single model
    print(f"\nModel Comparison:")
    print(f"  Best Single Model Accuracy: {best_fold_acc:.4f}")
    print(f"  Ensemble Model Accuracy: {ensemble_acc:.4f}")

    # Report improvement or decline
    diff = ensemble_acc - best_fold_acc
    if diff > 0:
        print(f"  Ensemble improves accuracy by {diff:.4f} ({diff * 100:.2f}%)")
    else:
        print(f"  Ensemble reduces accuracy by {abs(diff):.4f} ({abs(diff) * 100:.2f}%)")


def main():
    """
    Main function for bird species classification training with stratified k-fold cross-validation.
    """
    print("Starting bird species classification training with stratified k-fold cross-validation")
    print(f"Configuration: {N_SPLITS} folds, {BATCH_SIZE} batch size, {EPOCHS} max epochs")
    print(f"Using ensemble model: {USE_ENSEMBLE}")

    # Check which folds need training
    already_trained, remaining_folds = check_fold_status(OUTPUT_DIR, N_SPLITS)

    # Get data only for remaining folds
    fold_generator = prepare_data(remaining_folds, False)

    # Track metrics across folds
    fold_accuracies = []
    fold_histories = []
    all_reports = []

    # Store models and data for ensemble
    fold_models = []
    val_datasets = []
    label_encoders = []
    input_dim = None

    # Train remaining folds
    for fold_idx, fold_data in fold_generator:
        model, history, accuracy, report, val_dataset, label_encoder = train_fold(
            fold_idx, fold_data, BATCH_SIZE, EPOCHS, OUTPUT_DIR
        )

        if model is None:
            continue

        # Store results
        fold_accuracies.append(accuracy)
        fold_histories.append(history)
        all_reports.append(report)

        # Store model and data for ensemble
        fold_models.append(model)
        val_datasets.append(val_dataset)
        label_encoders.append(label_encoder)

        # Capture input dimension for ensemble model
        if input_dim is None:
            input_dim = model.input_shape[1]

    # Load already trained models for ensemble
    print("\n===== Loading all trained models for ensemble =====")

    # Clear previous lists if this is a re-run with only ensemble creation
    if not remaining_folds:
        fold_models = []
        val_datasets = []
        label_encoders = []
        input_dim = None

    # Load all existing fold models (both newly trained and previously trained)
    for fold_idx in range(N_SPLITS):
        # Check if the model exists
        model_path = f'{OUTPUT_DIR}/species_classifier_fold{fold_idx + 1}_keras'
        if os.path.exists(model_path):
            model, val_dataset, label_encoder = load_trained_fold(
                fold_idx, OUTPUT_DIR, BATCH_SIZE
            )

            if model is None:
                print(f"Could not load model for fold {fold_idx + 1}")
                continue

            print(f"Successfully loaded model for fold {fold_idx + 1}")

            # Store for ensemble
            fold_models.append(model)
            val_datasets.append(val_dataset)
            label_encoders.append(label_encoder)

            # Capture input dimension for ensemble model
            if input_dim is None:
                input_dim = model.input_shape[1]

    # Show summary of loaded models
    print(f"\nLoaded {len(fold_models)} fold models for ensemble creation")

    # Summarize results across all folds
    if fold_accuracies:
        summarize_results(fold_accuracies, all_reports, OUTPUT_DIR)
        create_final_tflite_model(OUTPUT_DIR)

    # Create and evaluate ensemble model if we have multiple folds
    if len(fold_models) > 1 and USE_ENSEMBLE:
        best_fold_acc = max(fold_accuracies) if fold_accuracies else 0.0
        create_and_evaluate_ensemble(
            fold_models, val_datasets, label_encoders, input_dim, OUTPUT_DIR, best_fold_acc
        )
    elif len(fold_models) <= 1:
        print("\nERROR: Need at least 2 models to create an ensemble, but found only", len(fold_models))

    print("\nTraining complete!")


if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()