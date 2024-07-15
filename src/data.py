import os
import logging
import numpy as np
import music21.note as note21
import music21.chord as chord21
from typing import List
from collections import namedtuple
from music21 import converter, instrument, stream

CustomNote = namedtuple("CustomNote", ("pitch", "velocity", "duration", "delay"))


def parse_midi_files(path):
    scores = []

    # Iterate over all MIDI files in all subdirectories
    for root, dirs, files in os.walk(path):
        logging.debug(f"Loading MIDI files from '{root}'...")

        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                path = os.path.join(root, file)

                try:
                    midi = converter.parse(path)
                    scores.append(midi)
                except:
                    logging.warning(f"MIDI file '{path}' could not be parsed!")

    logging.info(f"Loaded {len(scores)} MIDI files.")
    return scores


def extract_notes(scores, inst_filter: instrument = None):
    all_notes = []
    note_set = set()

    for index, score in enumerate(scores):
        logging.debug(f"Extracting notes from score {index}/{len(scores)}...")

        # Filter for specific instrument
        parts = instrument.partitionByInstrument(score)
        parts = [part for part in parts if not inst_filter or isinstance(part.getInstrument(), inst_filter)]

        # Always use the first part
        notes = parts[0].recurse()

        # Differentiate between single notes and chords
        for note in notes:
            if isinstance(note, note21.Note):
                pitch = note.pitch.nameWithOctave
            elif isinstance(note, chord21.Chord):
                pitch = ":".join([pitch.nameWithOctave for pitch in note.pitches])
            else:
                continue

            custom_note = CustomNote(pitch=pitch, velocity=note.volume.velocity, duration=note.duration.quarterLength,
                                     delay=0)  # TODO: Add the delay between the current and previous note
            all_notes.append(custom_note)
            note_set.add(custom_note)
            # TODO: Add only pitch to note_set and use other params as separate input for model (input-shape: input_seq, 3)

    logging.debug(f"Extracted {len(all_notes)} notes ({len(note_set)} individual notes).")
    return all_notes, note_set


def create_sequences(all_notes, note_set, seq_len: int):
    logging.debug("Creating sequences...")
    note_map = {note: index for index, note in enumerate(note_set)}
    input_data = []
    output_data = []

    # Create sequences with inputs and outputs
    for i in range(0, len(all_notes) - seq_len):
        sequence_in = all_notes[i:i + seq_len]
        sequence_out = all_notes[i + seq_len]
        input_data.append([note_map[note] for note in sequence_in])
        output_data.append(note_map[sequence_out])

    # Convert lists to one-hot encoding
    # input_data = [keras.utils.to_categorical(seq, num_classes=len(note_map)) for seq in input_data]
    # output_data = keras.utils.to_categorical(output_data, num_classes=len(note_map))

    input_data, output_data = np.array(input_data), np.array(output_data)
    logging.debug(f"Created input-sequences: {input_data.shape}, output-sequences: {output_data.shape} "
                  f"and value-map: ({len(note_map)}).")
    return input_data, output_data, note_map


def build_midi_file(notes: List[CustomNote], path):
    logging.debug("Building MIDI file...")
    note_output = []

    # Differentiate between single notes and chords
    for note in notes:
        if ":" in note.pitch:
            new_note = chord21.Chord(note.pitch.split(":"))
        else:
            new_note = note21.Note(note.pitch)

        new_note.volume.velocity = note.velocity
        new_note.duration.quarterLength = note.duration
        # TODO: Add delay
        new_note.storedInstrument = instrument.Piano()
        note_output.append(new_note)

    midi_stream = stream.Stream(note_output)
    midi_stream.write("midi", path)
    logging.debug(f"Saved MIDI file with {len(notes)} notes.")
