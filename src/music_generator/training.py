import pickle
import logging
from pathlib import Path

from music_generator.model import GenerationModel
from music_generator.utils import parse_midi_files, extract_notes, create_sequences

MIDI_IN_PATH = Path("./data/midi_in")
CACHE_PATH = Path("./data/cache")
MODEL_PATH = Path("./models")


def prepare_genres(genres: list[str], seq_len: int) -> None:
    """Parse MIDI files and cache note sets and training sequences."""
    for genre in genres:
        logging.info("Preparing genre '%s'...", genre)
        note_set_path = CACHE_PATH / f"{genre} - note_set.pkl"
        all_notes_path = CACHE_PATH / f"{genre} - all_notes.pkl"
        input_data_path = CACHE_PATH / f"{genre} - input_data.pkl"
        output_data_path = CACHE_PATH / \
            f"{genre} - output_data.pkl"
        note_map_path = CACHE_PATH / f"{genre} - note_map.pkl"

        if note_set_path.is_file() and all_notes_path.is_file():
            with note_set_path.open("rb") as file:
                note_set = pickle.load(file)

            with all_notes_path.open("rb") as file:
                all_notes = pickle.load(file)

            logging.info("Loaded processed MIDI files.")
        else:
            logging.info("Processing MIDI files.")
            midi_data = parse_midi_files(MIDI_IN_PATH / genre)
            all_notes, note_set = extract_notes(midi_data)

            with note_set_path.open("wb") as file:
                pickle.dump(note_set, file)

            with all_notes_path.open("wb") as file:
                pickle.dump(all_notes, file)

        if input_data_path.is_file() and output_data_path.is_file() and note_map_path.is_file():
            logging.info("Sequences are already saved.")
        else:
            logging.info("Creating new sequences.")
            input_data, output_data, note_map = create_sequences(
                all_notes, note_set, seq_len=seq_len)

            with input_data_path.open("wb") as file:
                pickle.dump(input_data, file)

            with output_data_path.open("wb") as file:
                pickle.dump(output_data, file)

            with note_map_path.open("wb") as file:
                pickle.dump(note_map, file)


def train_on_genre(
    genre: str,
    seq_len: int,
    embed_dims: int,
    batch_size: int,
    n_epochs: int,
    enable_logging: bool,
) -> None:
    """Train a model for a single genre and persist it with the note map."""
    input_data_path = CACHE_PATH / f"{genre} - input_data.pkl"
    output_data_path = CACHE_PATH / f"{genre} - output_data.pkl"
    note_map_path = CACHE_PATH / f"{genre} - note_map.pkl"

    if input_data_path.is_file() and output_data_path.is_file() and note_map_path.is_file():
        logging.info("Loading files for genre '%s'...", genre)

        with input_data_path.open("rb") as file:
            input_data = pickle.load(file)

        with output_data_path.open("rb") as file:
            output_data = pickle.load(file)

        with note_map_path.open("rb") as file:
            note_map = pickle.load(file)

        model = GenerationModel(logging=enable_logging)
        model.create(input_dim=len(note_map),
                     embed_dims=embed_dims, seq_len=seq_len)
        model.train(input_data=input_data, output_data=output_data,
                    batch_size=batch_size, n_epochs=n_epochs)

        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        timestamp = model.save(MODEL_PATH)

        with (MODEL_PATH / timestamp / "note_map.pkl").open("wb") as file:
            pickle.dump(note_map, file)

        logging.info("Finished training for model '%s'.", timestamp)
    else:
        logging.error(
            "Can't find files for genre '%s'. Run prepare first.", genre)
