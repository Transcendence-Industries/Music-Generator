import pickle
import logging
import numpy as np
from pathlib import Path

from music_generator.utils import CustomNote
from music_generator.model import GenerationModel

CACHE_PATH = Path("./data/cache")
MODEL_PATH = Path("./models")


def random_input_sequence(genre: str) -> np.ndarray:
    """Chooses random input sequence from training data."""
    input_data_path = CACHE_PATH / f"{genre} - input_data.pkl"

    with open(input_data_path, "rb") as file:
        input_data = pickle.load(file)

    input_seq = input_data[np.random.choice(input_data.shape[0]), :]
    return input_seq


def generate_with_model(
    model_name: str,
    input_seq: np.ndarray,
    n_values: int,
) -> tuple[list[CustomNote], list[CustomNote]]:
    """Generate a specific number of new notes for a given input sequence using a given model."""
    model_path = MODEL_PATH / model_name

    if model_path.is_dir():
        logging.info("Loading model '%s'...", model_name)

        with (model_path / "note_map.pkl").open("rb") as file:
            note_map = pickle.load(file)

        model = GenerationModel(logging=False)
        model.load(model_path)
        notes = model.predict(input_seq=input_seq, n_values=n_values)

        note_map = dict(zip(note_map.values(), note_map.keys()))
        new_notes = [note_map[note] for note in notes]
        input_notes = [note_map[note] for note in input_seq]
        logging.info("Finished predicting %s notes.", n_values)
        return input_notes, new_notes

    logging.error("Can't find model '%s'!", model_name)
    return None
