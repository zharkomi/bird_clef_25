#!/usr/bin/env python3
"""
Script to download Python packages from filtered requirements and pack them into a tar.gz file.
No command line arguments needed - uses hardcoded filenames.
"""

import os
import sys
import subprocess
import tarfile
import shutil
import tempfile
from datetime import datetime

# Configuration
FILTERED_REQUIREMENTS_FILE = 'requirements_filtered.txt'
DOWNLOAD_DIR = './downloaded_packages'
OUTPUT_FILE = f'python_packages_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tar.gz'

# Packages to explicitly exclude from download
EXCLUDED_PACKAGES = ['numpy']


def main():
    # Create download directory
    if os.path.exists(DOWNLOAD_DIR):
        print(f"Directory {DOWNLOAD_DIR} already exists. Cleaning it...")
        shutil.rmtree(DOWNLOAD_DIR)
    os.makedirs(DOWNLOAD_DIR)

    # Check if filtered requirements file exists
    if not os.path.isfile(FILTERED_REQUIREMENTS_FILE):
        print(f"Error: Filtered requirements file '{FILTERED_REQUIREMENTS_FILE}' not found.")
        print("Run detect_installed_packages.py first to generate it.")
        sys.exit(1)

    try:
        # Create a temporary filtered requirements file without excluded packages
        temp_req_file = tempfile.mktemp(suffix='.txt')
        excluded_count = 0

        try:
            with open(FILTERED_REQUIREMENTS_FILE, 'r') as f_in, open(temp_req_file, 'w') as f_out:
                for line in f_in:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name from the line
                        package_name = line.split('=')[0].split('>')[0].split('<')[0].split('~')[0].strip().lower()

                        # Check if package should be excluded
                        if package_name in [pkg.lower() for pkg in EXCLUDED_PACKAGES]:
                            print(f"Excluding package: {line}")
                            excluded_count += 1
                            continue

                        # Write line to the temporary file
                        f_out.write(line + '\n')

            # Count packages in temporary requirements file
            with open(temp_req_file, 'r') as f:
                package_count = sum(1 for line in f if line.strip() and not line.strip().startswith('#'))

            if excluded_count > 0:
                print(f"Excluded {excluded_count} packages from download.")

            if package_count == 0:
                print(
                    f"No packages to download after exclusions. All required packages are already installed or excluded.")
                print("Will only include wheel package in the archive.")
            else:
                print(f"Found {package_count} packages to download in filtered requirements.")

                # Download packages
                print(f"Downloading packages...")
                subprocess.run([
                    sys.executable, '-m', 'pip', 'download',
                    '-r', temp_req_file,
                    '--dest', DOWNLOAD_DIR,
                    '--no-binary', ':all:'
                ], check=True)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_req_file):
                os.remove(temp_req_file)

        # Also download wheel package (often needed for installation)
        print("Downloading wheel package...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'download',
            'wheel',
            '--dest', DOWNLOAD_DIR
        ], check=True)

        # Create tar.gz file
        print(f"Creating archive {OUTPUT_FILE}...")
        with tarfile.open(OUTPUT_FILE, 'w:gz') as tar:
            for file_name in os.listdir(DOWNLOAD_DIR):
                file_path = os.path.join(DOWNLOAD_DIR, file_name)
                arcname = os.path.basename(file_path)
                print(f"Adding {arcname} to archive...")
                tar.add(file_path, arcname=arcname)

        print(f"Archive created successfully: {OUTPUT_FILE}")

    except subprocess.CalledProcessError as e:
        print(f"Error during package download: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up
        if os.path.exists(DOWNLOAD_DIR):
            print(f"Cleaning up {DOWNLOAD_DIR}...")
            shutil.rmtree(DOWNLOAD_DIR)


if __name__ == '__main__':
    main()