# Libaries
from Constants import TUNED_KERAS
from Utilities import Utilities
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

class Classification:
    def __init__(self):
        self.utilities = Utilities()

    def is_trained(self):
        try:
            self.tuned_keras = load_model(TUNED_KERAS)
            return True
        except ValueError:
            return False

    def train_model(self, train_dataset):
        X, y = self.utilities.split_labels(train_dataset)

        self.tuned_keras = Sequential()
        self.tuned_keras.add(Dense(128, activation='relu', input_dim=300))
        self.tuned_keras.add(Dropout(0.3))
        self.tuned_keras.add(Dense(64, activation='relu'))
        self.tuned_keras.add(Dropout(0.3))
        self.tuned_keras.add(Dense(3, activation='softmax'))

        self.tuned_keras.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.tuned_keras.fit(X, y, epochs=150, batch_size=32, verbose=1)

        self.tuned_keras.save(TUNED_KERAS)

    def test_model(self, test_dataset):
        X, y = self.utilities.split_labels(test_dataset)

        loss, accuracy = self.tuned_keras.evaluate(X, y)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")