# Libraries
from Constants import DATASET
from Preprocessing import Preprocessing
from Classification import Classification
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    fasttext_model = Classification()
    # Load the dataset & normalization dictionary
    dataset = pd.read_csv(DATASET)

    # Preprocess the dataset
    preprocessing = Preprocessing(dataset)
    processed_dataset = preprocessing.execute()

    # Split Dataset into training and testing sets
    train_dataset, test_dataset = train_test_split(processed_dataset, test_size=0.2, random_state=42)

    for index, row in train_dataset.iterrows():
        if not row['teks'].strip():
            print(f"Train Data {index}: {row['teks']} - Label: {row['label']}")

    for index, row in test_dataset.iterrows():
        if not row['teks'].strip():
            print(f"Train Data {index}: {row['teks']} - Label: {row['label']}")
    
    # Train and test the model
    fasttext_model.train_model(train_dataset)
    fasttext_model.test_model(test_dataset)

if __name__ == "__main__":
    main()