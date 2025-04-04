# Example usage
from src import utils, birdnet
from src.predict import process_all_audio_files

if __name__ == "__main__":
    audio_directory = "/home/mikhail/prj/bc_25_data/train_soundscapes"
    output_file_path = "submission.csv"

    utils.TRAIN_DIR = "/home/mikhail/prj/bc_25_data/train_audio"
    birdnet.CSV_PATH = "/home/mikhail/prj/bc_25_data/taxonomy.csv"

    process_all_audio_files(audio_directory, output_file_path, batch_size=-1, dir_limit=50)
