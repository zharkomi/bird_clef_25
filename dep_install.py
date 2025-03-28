#!/usr/bin/env python3
"""
Module to unpack and install Python dependencies from a tar.gz file in a Kaggle notebook.
Import this module and call install_dependencies() in a Kaggle notebook to install offline packages.
"""

import os
import sys
import subprocess
import tarfile
import tempfile
import glob
import shutil


def install_dependencies(archive_path):
    """
    Unpack and install Python dependencies from a tar.gz archive.

    Args:
        archive_path (str): Path to the tar.gz archive containing Python packages

    Returns:
        bool: True if installation was successful, False otherwise
    """
    # Check if archive exists
    if not os.path.isfile(archive_path):
        print(f"Error: Archive file '{archive_path}' not found.")
        return False

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix='kaggle_pkg_')

    try:
        # Extract the archive
        print(f"Extracting {archive_path} to {temp_dir}...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)

        # Install wheel first (if present) to ensure we can install other packages
        wheel_files = glob.glob(os.path.join(temp_dir, "wheel*.tar.gz"))
        if wheel_files:
            print("Installing wheel package first...")
            for wheel_file in wheel_files:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '--no-index',
                    wheel_file
                ], check=True)

        # Install all extracted packages
        print("Installing packages...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', temp_dir,
            os.path.join(temp_dir, '*')
        ], check=True)

        print("All packages installed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during package installation: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            print(f"Cleaning up {temp_dir}...")
            shutil.rmtree(temp_dir)


# Example usage if run as a script
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python install_dependencies.py <archive_path>")
        sys.exit(1)

    success = install_dependencies(sys.argv[1])
    if not success:
        sys.exit(1)