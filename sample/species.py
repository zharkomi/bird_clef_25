import csv


def find_missing_species(labels_file, taxonomy_file):
    """
    Finds species in taxonomy.csv which are absent in the labels file.

    Args:
        labels_file: Path to BirdNET_GLOBAL_6K_V2.4_Labels.txt
        taxonomy_file: Path to taxonomy.csv

    Returns:
        List of primary_labels for species not found in the labels file
    """
    # Step 1: Extract scientific names and common names from labels file
    scientific_names = set()
    common_names = set()

    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('_', 1)
                if len(parts) == 2:
                    scientific_name, common_name = parts
                    scientific_names.add(scientific_name.lower())
                    common_names.add(common_name.lower())

    # Step 2: Check taxonomy entries against both sets
    missing_species = []

    with open(taxonomy_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scientific_name = row['scientific_name'].lower() if row['scientific_name'] else ''
            common_name = row['common_name'].lower() if row['common_name'] else ''

            # If neither scientific nor common name is in the labels file
            if scientific_name not in scientific_names and common_name not in common_names:
                missing_species.append({
                    'primary_label': row['primary_label'],
                    'scientific_name': row['scientific_name'],
                    'common_name': row['common_name']
                })

    return missing_species


# Main execution
if __name__ == "__main__":
    LABELS_FILE = "../bn/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
    CSV_PATH = "/home/mikhail/prj/bc_25_data/taxonomy.csv"

    missing = find_missing_species(LABELS_FILE, CSV_PATH)

    # Print results
    print(f"Found {len(missing)} species in taxonomy that are not in the labels file:")
    for species in missing:
        print(
            f"\"{species['primary_label']}\",")

    # Option to save results to a file
    with open("../missing_species.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["primary_label", "scientific_name", "common_name"])
        writer.writeheader()
        writer.writerows(missing)

    print(f"\nResults saved to missing_species.csv")