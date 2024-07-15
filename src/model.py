import os
import keras
import mlflow
import numpy as np
from datetime import datetime
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense

from logger import MLFlow_Logger


class MusicGen_Model:
    def __init__(self, logging=True):
        self.logging = logging
        self.model = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.seq_len = None
        self.embed_dims = None

        if logging:
            self.logger = MLFlow_Logger(experiment=self.__class__.__name__)

    def create(self, input_dim: int, embed_dims: int, seq_len: int):
        self.seq_len = seq_len
        self.embed_dims = embed_dims

        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_dim, output_dim=embed_dims, input_length=seq_len))
        self.model.add(LSTM(units=128, return_sequences=True))
        self.model.add(LSTM(units=128, return_sequences=False))
        self.model.add(Dense(units=input_dim, activation="softmax"))
        self.model.summary()

    def load(self, path: str):
        self.model = keras.models.load_model(path)
        self.model.summary()

    def save(self, path: str):
        self.model.save(os.path.join(path, self.timestamp))
        return self.timestamp

    def train(self, input_data: np.array, output_data: np.array, batch_size: int, n_epochs: int):
        self.logger.create_run(run=self.timestamp)
        self.logger.log_parameters({"CUSTOM n_episodes": n_epochs,
                                    "CUSTOM batch_size": batch_size,
                                    "CUSTOM seq_len": self.seq_len,
                                    "CUSTOM embed_dims": self.embed_dims})

        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(input_data, output_data, epochs=n_epochs, batch_size=batch_size,
                       callbacks=[mlflow.tensorflow.MLflowCallback()])
        self.logger.end_run()

    def predict(self, input_seq: np.array, n_values: int):
        pred_values = []

        for _ in range(n_values):
            input_seq = np.reshape(input_seq, (1, len(input_seq)))
            pred_value = self.model.predict(input_seq)
            pred_value = np.argmax(pred_value)
            pred_values.append(pred_value)
            input_seq = np.append(input_seq[0][1:], pred_value)

        return np.array(pred_values)
