# Libaries
from Constants import PRETRAINED_MODEL, NORMALISATION, STOPWORDS
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import fasttext

class Cleaner:
    def __init__(self):
        pass

    def clean_url(self, text):
        # Clean URL that starts with http, https, www, or domain names
        text = re.sub(r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', text)
        # Clean URL with query parameters or fragments
        text = re.sub(r'http\S+(\?\S*)?|www\.\S+(\?\S*)?', ' ', text)
        # Clean URL with anchors
        text = re.sub(r'http\S+(\#\S*)?|www\.\S+(\#\S*)?', ' ', text)
        # Clean URLs with ports
        text = re.sub(r'https?://[a-zA-Z0-9.-]+(:\d+)?(/[^\s]*)?', ' ', text)
        return text
    
    def clean_hashtag(self, text):
        # Clean hashtags
        text = re.sub(r'#\w+', ' ', text)
        return text
    
    def clean_email(self, text):
        # Clean email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' ', text)
        return text
    
    def clean_number(self, text):
        # Clean numbers
        text = re.sub(r'\d+', ' ', text)
        return text
    
    def clean_punctuation(self, text):
        # Clean punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        return text
    
    def clean_extra_spaces(self, text):
        # Clean extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def clean_repeated_words(self, text):
        # Clean repeated words
        text = re.sub(r'(\b\w+\b)( \1)+', r'\1', text)
        return text 

class Preprocessing:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.normalization_dict = pd.read_csv(NORMALISATION)
        self.stopword_dict = pd.read_csv(STOPWORDS)
        self.pretrained_model = fasttext.load_model(PRETRAINED_MODEL)

    def fix_wording(self, text):
        words = text.split()
        fixed_words = []
        for word in words:
            if self.pretrained_model.get_word_id(word) == 0:
                suggestions = self.pretrained_model.get_nearest_neighbors(word)
                if suggestions:
                    fixed_words.append(suggestions[0][1])
                else:
                    fixed_words.append(word)
            else:
                fixed_words.append(word)
        return ' '.join(fixed_words)

    def case_folding(self, text):
        return text.lower()

    def cleaning(self, text):
        cleaner = Cleaner()
        text = cleaner.clean_url(text)
        text = cleaner.clean_hashtag(text)
        text = cleaner.clean_number(text)
        text = cleaner.clean_punctuation(text)
        text = cleaner.clean_extra_spaces(text)
        text = cleaner.clean_repeated_words(text)
        return text

    def normalisation(self, text):
        norm_dict = dict(zip(self.normalization_dict['singkat'], self.normalization_dict['hasil']))
        words = text.split()
        normalized_words = [norm_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)

    def stemming(self, text):
        stemmer = StemmerFactory().create_stemmer()
        return stemmer.stem(text)

    def stopword_removal(self, text):
        stopwords = set(self.stopword_dict['stopword'].tolist())
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords]
        return ' '.join(filtered_words)

    def execute(self):
        for index, row in self.dataset.iterrows():
            text = str(row['teks'])
            text = self.fix_wording(text)
            text = self.case_folding(text)
            text = self.cleaning(text)
            text = self.normalisation(text)
            text = self.stemming(text)
            text = self.stopword_removal(text)
            self.dataset.at[index, 'teks'] = text
        return self.dataset