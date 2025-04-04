import os
import numpy as np
import pandas as pd
import librosa
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer


# Function to get BirdNET analyzer instance
def get_analyzer():
    # Initialize the analyzer with default settings
    # The BirdNET model expects 48kHz audio
    return Analyzer(sampling_rate=48000)


# Function to analyze audio chunk and get embeddings
def analyze_chunk(chunk, sample_rate):
    # Create a RecordingBuffer object for this chunk
    # RecordingBuffer is designed to work with raw audio buffers
    analyzer = get_analyzer()
    recording = RecordingBuffer(
        analyzer,
        buffer=chunk,
        rate=sample_rate,
        return_all_detections=True
    )
    # Process the audio data
    recording.analyze()
    return recording


# Function to extract embeddings from an audio file
def extract_embeddings(audio_path):
    try:
        # Load audio file at original sample rate
        audio, sr = librosa.load(audio_path, sr=None)

        # Resample to 48000 Hz (BirdNET requirement)
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000

        # Calculate chunk size in samples (5 seconds * sample rate)
        chunk_size = 5 * sr

        # Get total number of complete chunks
        num_chunks = len(audio) // chunk_size

        # Storage for embeddings from all chunks
        all_embeddings = []

        # Process each 5-second chunk
        for i in range(num_chunks):
            # Extract the chunk
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = audio[start_idx:end_idx]

            # Analyze the chunk
            recording = analyze_chunk(chunk, sr)

            # Get embeddings from the recording
            # Note: This assumes that birdnetlib RecordingBuffer provides access to embeddings
            # You might need to modify this part based on the actual API
            chunk_embeddings = recording.embeddings

            # Store embeddings if valid
            if chunk_embeddings is not None and len(chunk_embeddings) > 0:
                all_embeddings.append(chunk_embeddings)

        # If no embeddings were extracted from any chunk, return None
        if not all_embeddings:
            print(f"Warning: No embeddings extracted for {audio_path}")
            return None

        # Flatten list of embeddings from all chunks
        flat_embeddings = np.vstack(all_embeddings)

        # Aggregate embeddings (e.g., average them)
        aggregated_embedding = np.mean(flat_embeddings, axis=0)

        return aggregated_embedding

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


# Function to collect all audio files by species
def collect_audio_files(base_dir):
    species_files = {}

    # Iterate through each species directory
    for species in os.listdir(base_dir):
        species_dir = os.path.join(base_dir, species)
        if os.path.isdir(species_dir):
            audio_files = []
            # Collect only OGG files for the current species
            for file in os.listdir(species_dir):
                if file.lower().endswith('.ogg'):
                    audio_files.append(os.path.join(species_dir, file))

            if audio_files:  # Only add species if it has audio files
                species_files[species] = audio_files
                print(f"Found {len(audio_files)} OGG files for species: {species}")

    return species_files


# Function to split data into train and test sets
def split_data(species_files, test_size=0.2, random_state=42, min_samples=2):
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []

    # Keep track of excluded species
    excluded_species = []

    # For each species, split its files into train and test if it has enough samples
    for species, files in species_files.items():
        if len(files) < min_samples:
            print(
                f"WARNING: Species '{species}' has only {len(files)} sample(s), which is less than the minimum {min_samples} required. Excluding from dataset.")
            excluded_species.append(species)
            continue

        species_train, species_test = train_test_split(
            files, test_size=test_size, random_state=random_state
        )

        train_files.extend(species_train)
        test_files.extend(species_test)
        train_labels.extend([species] * len(species_train))
        test_labels.extend([species] * len(species_test))

    if excluded_species:
        print(f"Excluded {len(excluded_species)} species due to insufficient samples: {', '.join(excluded_species)}")

    # Check if we have any data left
    if not train_files or not test_files:
        raise ValueError("After filtering species with insufficient samples, no data remains for training/testing.")

    return train_files, test_files, train_labels, test_labels


# Function to prepare dataset for TensorFlow
def prepare_dataset(file_paths, labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Pre-extract all embeddings
    embeddings = []
    valid_indices = []
    embedding_dim = None

    print(f"Extracting embeddings from {len(file_paths)} files...")
    for i, file_path in enumerate(file_paths):
        if i % 20 == 0:  # Print progress every 20 files
            print(f"Processing file {i + 1}/{len(file_paths)}")

        embedding = extract_embeddings(file_path)
        if embedding is not None:
            embeddings.append(embedding)
            valid_indices.append(i)

            # Determine embedding dimension if not yet set
            if embedding_dim is None:
                embedding_dim = embedding.shape[0]

    # If no valid embeddings were found, use a default dimension
    if not embeddings:
        print("Warning: No valid embeddings extracted from any files")
        return None, None, label_encoder, 0

    # Filter labels to only include those with valid embeddings
    filtered_labels = [labels[i] for i in valid_indices]
    encoded_labels = label_encoder.fit_transform(filtered_labels)

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)

    print(f"Successfully extracted embeddings from {len(embeddings)} files")
    print(f"Embedding dimension: {embedding_dim}")

    return embeddings, encoded_labels, label_encoder, len(embeddings)


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
def train_model(train_dataset, val_dataset, input_dim, num_classes, epochs=20):
    # Create model
    model = create_model(input_dim, num_classes)

    # Define callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        'best_species_classifier.h5',
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
def evaluate_model(model, test_dataset, label_encoder):
    # Predict on test dataset
    predictions = []
    true_labels = []

    for embeddings, labels in test_dataset:
        batch_predictions = model.predict(embeddings)
        predictions.extend(np.argmax(batch_predictions, axis=1))
        true_labels.extend(labels.numpy())

    # Convert numeric labels back to original species names
    pred_species = label_encoder.inverse_transform(predictions)
    true_species = label_encoder.inverse_transform(true_labels)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_species, pred_species))

    # Print overall accuracy
    accuracy = accuracy_score(true_species, pred_species)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Create confusion matrix
    species_names = label_encoder.classes_
    print("\nConfusion Matrix:")
    cm = pd.crosstab(
        pd.Series(true_species, name='Actual'),
        pd.Series(pred_species, name='Predicted')
    )
    print(cm)

    return pred_species, true_species


# Main function to run the entire pipeline
def main(base_dir, batch_size=32, test_size=0.2, val_size=0.2, random_state=42, min_samples=2):
    # Step 1: Collect audio files for each species
    print("Collecting audio files...")
    species_files = collect_audio_files(base_dir)

    # Print summary of collected files
    print(f"Found {len(species_files)} species with audio files:")
    for species, files in species_files.items():
        print(f"  - {species}: {len(files)} files")

    # Step 2: Split data into train and test sets
    print("\nSplitting data into train and test sets...")
    try:
        train_val_files, test_files, train_val_labels, test_labels = split_data(
            species_files, test_size=test_size, random_state=random_state, min_samples=min_samples
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Aborting training process.")
        return

    # Check if we have classes with very few samples in the validation split
    try:
        # Further split train into train and validation
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels, test_size=val_size, random_state=random_state,
            stratify=train_val_labels
        )
    except ValueError as e:
        print(f"Warning: {e}")
        print("Some classes have too few samples for stratified validation split.")
        print("Falling back to non-stratified validation split.")

        # Fall back to non-stratified split for validation
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels, test_size=val_size, random_state=random_state,
            stratify=None
        )

    print(f"Train set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Test set: {len(test_files)} files")

    # Step 3: Prepare datasets
    print("\nPreparing training dataset...")
    train_embeddings, train_encoded_labels, label_encoder, train_size = prepare_dataset(train_files, train_labels)

    print("\nPreparing validation dataset...")
    val_embeddings, val_encoded_labels, _, val_size = prepare_dataset(val_files, val_labels)

    print("\nPreparing test dataset...")
    test_embeddings, test_encoded_labels, _, test_size = prepare_dataset(test_files, test_labels)

    # Check if we have enough data to continue
    if train_size == 0 or val_size == 0 or test_size == 0:
        print("Error: Not enough valid embeddings extracted to train the model.")
        return

    # Get input dimension and number of classes
    input_dim = train_embeddings.shape[1]
    num_classes = len(label_encoder.classes_)

    # Step 4: Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_embeddings, train_encoded_labels, batch_size, is_training=True)
    val_dataset = create_tf_dataset(val_embeddings, val_encoded_labels, batch_size)
    test_dataset = create_tf_dataset(test_embeddings, test_encoded_labels, batch_size)

    # Step 5: Train the model
    print("\nTraining model...")
    model, history = train_model(train_dataset, val_dataset, input_dim, num_classes)

    # Step 6: Evaluate the model
    print("\nEvaluating model on test set...")
    pred_species, true_species = evaluate_model(model, test_dataset, label_encoder)

    # Step 7: Save the model and label encoder
    print("\nSaving model and label encoder...")
    model.save('species_classifier_final.h5')
    with open('species_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Plot accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
    except:
        print("Could not create training history plot. Matplotlib may not be installed.")

    print("Training and evaluation complete!")


# Function to use the trained model for prediction
def predict_species(audio_file, model_path='species_classifier_final.h5', encoder_path='species_label_encoder.pkl'):
    # Load the model and label encoder
    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Check if file is OGG format
    if not audio_file.lower().endswith('.ogg'):
        print(f"Warning: {audio_file} is not an OGG file. Model was trained on OGG files only.")

    # Extract embeddings from the audio file
    embedding = extract_embeddings(audio_file)

    if embedding is None:
        print(f"Could not extract embeddings from {audio_file}")
        return None

    # Reshape embedding for prediction
    embedding = np.expand_dims(embedding, axis=0)

    # Make prediction
    prediction = model.predict(embedding)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Convert to species name
    species_name = label_encoder.inverse_transform([predicted_class])[0]

    return {
        'species': species_name,
        'confidence': float(confidence),
        'all_probabilities': {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(prediction)
        }
    }


if __name__ == "__main__":
    # Replace with the path to your base directory containing species folders
    base_dir = "/home/mikhail/prj/bc_25_data/train_audio"
    main(base_dir, min_samples=2)  # Require at least 2 samples per species