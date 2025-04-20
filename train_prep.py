import glob
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


EMBEDDINGS_DIR = '/home/mikhail/prj/bird_clef_25/embeddings'
N_SPLITS = 5
RANDOM_STATE = 42


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


# Helper function to load and yield fold data from disk
def load_fold_from_disk(fold_idx, train_dir):
    """
    Loads fold data from a pickle file and yields it if available.

    Args:
        fold_idx: Index of the fold to load
        train_dir: Directory where fold files are stored

    Returns:
        fold_data if file exists, None otherwise
    """
    fold_file = os.path.join(train_dir, f"fold_{fold_idx + 1}.pkl")
    model_file = os.path.join(train_dir, f"species_classifier_fold{fold_idx + 1}.keras")

    if os.path.exists(model_file):
        print(f"Fold {fold_idx + 1} already trained. Skipping.")
        return None

    if os.path.exists(fold_file):
        print(f"Loading saved data for fold {fold_idx + 1} from {fold_file}")
        with open(fold_file, "rb") as f:
            fold_data = pickle.load(f)
        return fold_data
    else:
        print(f"Warning: Data for fold {fold_idx + 1} not found at {fold_file}")
        return None


def check_fold_status(output_dir, n_splits):
    """
    Check which folds are already trained and which need training.

    Args:
        n_splits: Total number of folds to check

    Returns:
        tuple: (already_trained, remaining_folds) lists of fold indices
    """
    os.makedirs(output_dir, exist_ok=True)

    already_trained = []
    remaining_folds = []

    for fold_idx in range(n_splits):
        model_file = os.path.join(output_dir, f"best_species_classifier_fold{fold_idx + 1}.keras")
        fold_file = os.path.join(output_dir, f"fold_{fold_idx + 1}.pkl")

        if os.path.exists(model_file):
            print(f"Fold {fold_idx + 1} already trained. Skipping.")
            already_trained.append(fold_idx)
        elif os.path.exists(fold_file):
            print(f"Fold {fold_idx + 1} has prepared data but not trained yet.")
            remaining_folds.append(fold_idx)
        else:
            print(f"Fold {fold_idx + 1} needs data preparation and training.")
            remaining_folds.append(fold_idx)

    print(f"Already trained folds: {already_trained}")
    print(f"Remaining folds to train: {remaining_folds}")

    return already_trained, remaining_folds


# Modified prepare_data function
def prepare_data(folds_to_train=None):
    """
    Prepares data for training. Saves all fold data to disk first,
    then cleans memory, and finally yields folds read from disk one by one.

    Args:
        folds_to_train: List of fold indices to prepare data for. If None, all folds are prepared.

    Yields:
        fold_idx, fold_data: Index of the fold and the corresponding training data
    """
    train_dir = "train"
    os.makedirs(train_dir, exist_ok=True)

    # If we're continuing training with specific folds, load only those fold files
    if folds_to_train is not None:
        print(f"Loading specified folds for training: {folds_to_train}")
        for fold_idx in folds_to_train:
            fold_data = load_fold_from_disk(fold_idx, train_dir)
            if fold_data is not None:
                yield fold_idx, fold_data

        # End function after processing requested folds
        return

    # If no specific folds are requested, proceed with normal preparation
    print("Collecting embedding files...")
    species_files = collect_embedding_files(EMBEDDINGS_DIR)
    print(f"Found {len(species_files)} species with embedding files:")
    for species, files in species_files.items():
        print(f"  - {species}: {len(files)} files")

    print(f"\nLoading all embeddings and preparing {N_SPLITS}-fold stratified cross-validation...")
    all_embeddings, all_labels, kfold, stratify_labels = prepare_kfold_datasets(
        species_files, n_splits=N_SPLITS, random_state=RANDOM_STATE
    )

    # First step: Save all folds to disk
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_embeddings, stratify_labels)):
        fold_file = os.path.join(train_dir, f"fold_{fold_idx + 1}.pkl")
        model_file = os.path.join(train_dir, f"species_classifier_fold{fold_idx + 1}.keras")

        if os.path.exists(model_file):
            print(f"Fold {fold_idx + 1} already trained. Skipping fold preparation.")
            continue

        print(f"Preparing and saving data for fold {fold_idx + 1}...")
        fold_data = prepare_fold_data(all_embeddings, all_labels, train_idx, val_idx)
        with open(fold_file, "wb") as f:
            pickle.dump(fold_data, f)
        print(f"Fold {fold_idx + 1} data saved to {fold_file}")

    # Second step: Clean memory
    print("Clearing memory...")
    del all_embeddings
    del all_labels
    del kfold
    del stratify_labels
    import gc
    gc.collect()
    print("Memory cleared")

    # Third step: Yield folds one by one from disk
    print("Beginning to yield folds from disk...")
    for fold_idx in range(N_SPLITS):
        fold_data = load_fold_from_disk(fold_idx, train_dir)
        if fold_data is not None:
            yield fold_idx, fold_data
