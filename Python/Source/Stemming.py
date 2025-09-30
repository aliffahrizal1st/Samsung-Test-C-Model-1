# Libaries
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Stemming:
    def __init__(self, dataset, normalization_dict):
        self.dataset = dataset
        self.normalization_dict = normalization_dict
        self.stemmer = StemmerFactory().create_stemmer()

    def read_local_dictionary(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        dictionary = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                dictionary[parts[0]] = parts[1]
        return dictionary
        
    def stemming(self):
        for index, row in self.dataset.iterrows():
            text = row['teks']
            self.stemmer.stem(text)
            self.dataset.at[index, 'teks'] = text

    def executed(self):
        self.stemming()
            