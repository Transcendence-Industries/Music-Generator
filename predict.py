import os
import pickle
import logging
import numpy as np

from data import build_midi_file
from model import MusicGen_Model

MIDI_OUTPUT_PATH = "./midi_out"
DATA_PATH = "./data"
MODEL_PATH = "./models"


def predict_with_model(model_name, input_seq, n_values):
    model_path = os.path.join(MODEL_PATH, model_name)

    if os.path.isdir(model_path):
        logging.info(f"Loading model '{model_name}'...")

        # Load note-map from model
        with open(os.path.join(model_path, "note_map.pkl"), "rb") as f:
            note_map = pickle.load(f)

        # Build and predict with model
        model = MusicGen_Model(logging=False)
        model.load(model_path)
        notes = model.predict(input_seq=input_seq, n_values=n_values)

        # Decode to notes
        note_map = dict(zip(note_map.values(), note_map.keys()))
        new_notes = [note_map[note] for note in notes]
        input_notes = [note_map[note] for note in input_seq]
        logging.info(f"Finished predicting {n_values} notes.")
        return input_notes, new_notes
    else:
        logging.error(f"Can't find model '{model_name}'!")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    genre = "Ambient"

    # Choose random input sequence from training data
    input_data_path = os.path.join(DATA_PATH, f"{genre} - input_data.pkl")
    with open(input_data_path, "rb") as file:
        input_data = pickle.load(file)
    input_s = input_data[np.random.choice(input_data.shape[0]), :]

    # Predict new notes and build MIDI file
    input_s, predictions = predict_with_model(model_name="2024-10-16_19-23-36", input_seq=input_s, n_values=100)
    complete_s = input_s + predictions
    build_midi_file(complete_s, os.path.join(MIDI_OUTPUT_PATH, "output.mid"))
