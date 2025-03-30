# Example usage
from src.predict import process_all_audio_files

if __name__ == "__main__":
    workd_dir = "/home/mikhail/prj/bc_25_data/work/"
    audio_directory = "/home/mikhail/prj/bc_25_data/small_soundscapes_1"
    train_directory = "/home/mikhail/prj/bc_25_data/train_audio"
    species_csv_path = "/home/mikhail/prj/bc_25_data/taxonomy.csv"
    sample_csv_path = "/home/mikhail/prj/bc_25_data/sample_submission.csv"
    output_file_path = "submission.csv"

    process_all_audio_files(workd_dir,
                            audio_directory,
                            species_csv_path,
                            output_file_path,
                            train_directory,
                            dir_limit=10)
