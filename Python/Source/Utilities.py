# Libaries
from Constants import PRETRAINED_MODEL
import fasttext
import numpy as np
from keras.utils import to_categorical

class Utilities:
    def __init__(self):
        self.fasttext_model = fasttext.load_model(PRETRAINED_MODEL)

    def embeddings(self, text):
        words = text.split()
        if not words:
            return np.zeros(self.fasttext_model.get_dimension())
        embeddings = [self.fasttext_model.get_word_vector(word) for word in words]
        return np.mean(embeddings, axis=0)
    
    def split_labels(self, dataset):
        X = dataset['teks'].apply(self.embeddings).to_list()
        X = np.vstack(X)
        y = dataset['label']
        return X, y
