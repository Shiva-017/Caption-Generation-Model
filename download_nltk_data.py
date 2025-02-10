# download_nltk_data.py

import nltk

def download_punkt():
    nltk.download('punkt')

if __name__ == "__main__":
    download_punkt()
    print("NLTK 'punkt' tokenizer data downloaded successfully.")
