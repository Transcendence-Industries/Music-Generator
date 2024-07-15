import os
import pickle
import logging

from data import parse_midi_files, extract_notes, create_sequences
from model import MusicGen_Model

MIDI_INPUT_PATH = "./midi_in"
CACHE_PATH = "./cache"
DATA_PATH = "./data"
MODEL_PATH = "./models"


def prepare_genres(genres, seq_len):
    for genre in genres:
        logging.info(f"Preparing genre '{genre}'...")
        note_set_path = os.path.join(CACHE_PATH, f"{genre} - note_set.pkl")
        all_notes_path = os.path.join(CACHE_PATH, f"{genre} - all_notes.pkl")
        input_data_path = os.path.join(DATA_PATH, f"{genre} - input_data.pkl")
        output_data_path = os.path.join(DATA_PATH, f"{genre} - output_data.pkl")
        note_map_path = os.path.join(DATA_PATH, f"{genre} - note_map.pkl")

        # Prepare midi data
        if os.path.isfile(note_set_path) and os.path.isfile(all_notes_path):
            with open(note_set_path, "rb") as f:
                note_set = pickle.load(f)

            with open(all_notes_path, "rb") as f:
                all_notes = pickle.load(f)

            logging.info("Loaded processed MIDI files.")
        else:
            logging.info("Processing MIDI files.")
            midi_data = parse_midi_files(os.path.join(MIDI_INPUT_PATH, genre))
            all_notes, note_set = extract_notes(midi_data)

            with open(note_set_path, "wb") as f:
                pickle.dump(note_set, f)

            with open(all_notes_path, "wb") as f:
                pickle.dump(all_notes, f)

        # Create sequences
        if os.path.isfile(input_data_path) and os.path.isfile(output_data_path) and os.path.isfile(note_map_path):
            logging.info("Sequences are already saved.")
        else:
            logging.info("Creating new sequences.")
            input_data, output_data, note_map = create_sequences(all_notes, note_set, seq_len=seq_len)

            with open(input_data_path, "wb") as f:
                pickle.dump(input_data, f)

            with open(output_data_path, "wb") as f:
                pickle.dump(output_data, f)

            with open(note_map_path, "wb") as f:
                pickle.dump(note_map, f)


def train_on_genre(genre, seq_len, embed_dims, batch_size, n_epochs):
    input_data_path = os.path.join(DATA_PATH, f"{genre} - input_data.pkl")
    output_data_path = os.path.join(DATA_PATH, f"{genre} - output_data.pkl")
    note_map_path = os.path.join(DATA_PATH, f"{genre} - note_map.pkl")

    if os.path.isfile(input_data_path) and os.path.isfile(output_data_path) and os.path.isfile(note_map_path):
        logging.info(f"Loading files for genre '{genre}'...")

        # Load data
        with open(input_data_path, "rb") as f:
            input_data = pickle.load(f)

        with open(output_data_path, "rb") as f:
            output_data = pickle.load(f)

        with open(note_map_path, "rb") as f:
            note_map = pickle.load(f)

        # Build and train model
        model = MusicGen_Model()
        model.create(input_dim=len(note_map), embed_dims=embed_dims, seq_len=seq_len)
        model.train(input_data=input_data, output_data=output_data, batch_size=batch_size, n_epochs=n_epochs)
        timestamp = model.save(MODEL_PATH)

        # Add note-map to model
        with open(os.path.join(MODEL_PATH, timestamp, "note_map.pkl"), "wb") as f:
            pickle.dump(note_map, f)

        logging.info(f"Finished training for model '{timestamp}'.")
    else:
        logging.error(f"Can't find files for genre '{genre}'!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    all_genres = ("Ambient", "Country", "Latin", "Religious", "Unknown", "Blues", "Electronic", "Pop", "Rock", "World",
                  "Children", "Folk", "Rap", "Soul", "Classical", "Jazz", "Reggae", "Soundtracks")

    # Prepare sequences and train model
    prepare_genres(genres=all_genres, seq_len=100)
    train_on_genre(genre="Ambient", seq_len=100, embed_dims=64, batch_size=64, n_epochs=10)
