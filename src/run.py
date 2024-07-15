import logging
import argparse

from music_generator.utils import build_midi_file
from music_generator.training import prepare_genres, train_on_genre
from music_generator.inference import random_input_sequence, generate_with_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Music Generator controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    training_parser = subparsers.add_parser(
        "training", help="Prepare data and train a model")
    training_parser.add_argument(
        "--genre", required=True, help="Genre subfolder")
    training_parser.add_argument(
        "--seq-len", type=int, default=100, help="Sequence length")
    training_parser.add_argument(
        "--embed-dims", type=int, default=64, help="Embedding dimensions")
    training_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size")
    training_parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs")
    training_parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow logging")

    inference_parser = subparsers.add_parser(
        "inference", help="Generate MIDI from a trained model")
    inference_parser.add_argument(
        "--genre", required=True, help="Genre subfolder")
    inference_parser.add_argument(
        "--model-name", required=True, help="Name of the trained model")
    inference_parser.add_argument(
        "--n-values", type=int, default=100, help="Number of notes to generate")

    args = parser.parse_args()

    if args.command == "training":
        prepare_genres(genres=[args.genre], seq_len=args.seq_len)
        train_on_genre(
            genre=args.genre,
            seq_len=args.seq_len,
            embed_dims=args.embed_dims,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            enable_logging=not args.no_mlflow,
        )
    elif args.command == "inference":
        input_s = random_input_sequence(genre=args.genre)
        output_s, predictions = generate_with_model(
            model_name=args.model_name,
            input_seq=input_s,
            n_values=args.n_values,
        )
        complete_s = output_s + predictions
        build_midi_file(complete_s, f"{args.genre}.mid")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
