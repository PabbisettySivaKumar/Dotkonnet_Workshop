import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk  
from nltk.stem import WordNetLemmatizer
import fitz
import docx
import matplotlib.pyplot as plt
from textblob import TextBlob
import textstat
from sklearn.feature_extraction.text import CountVectorizer

def text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text
def text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    text = ""
    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text
def preprocess_text(text):
    """Preprocess text by tokenization, removing stopwords, and lemmatization."""
    lematizer= WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    processed_tokens = [
        lematizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return processed_tokens

def detect_keywords(text, top_n=10 ):
    vectorizer= CountVectorizer(stop_words='english',ngram_range=(1,2))
    X= vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    counts= X.sum(axis=0).A1
    word_count = dict(zip(feature_names, counts))
    sorted_keywords = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_keywords[:top_n]

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def calculate_readability(text):

    metrics= {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text)
    }
    return metrics

# ----------------------------
# 1. NLTK Setup
# ----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# ----------------------------
# 2. Main Execution
# ----------------------------

if __name__== "__main__":

    file_path= '/Users/sivakumar/Downloads/Trump_Putin_Summit_Articles.docx'  # Change to your file path
    if file_path.endswith('.pdf'):
        text_content= text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        text_content= text_from_docx(file_path)
    else:
        print("Unsupported file format. Please provide a PDF or DOCX file.")
        exit()

    if text_content.startswith("Error:"):
        print(text_content)
    else:
        preprocessed_tokens = preprocess_text(text_content)
        print('----------------------')
        print("Preprocessed Tokens:")
        print("----------------------\n")
        print(preprocessed_tokens[:50])

        keywords = detect_keywords(text_content)
        print("----------------------")
        print("Top Keywords:")
        print("----------------------")
        for keyword, count in keywords:
            print(f"{keyword}: {count}")

        sentiment= analyze_sentiment(text_content)
        print("----------------------")
        print("Sentiment Analysis:")
        print("----------------------")
        print(f"Sentiment Polarity: {sentiment}")

        if sentiment > 0.1:
            print("The text has a positive sentiment.")
        elif sentiment < -0.1:
            print("The text has a negative sentiment.")
        else:
            print("The text has a neutral sentiment.")

        redability_metrics= calculate_readability(text_content)
        print("----------------------")
        print("Readability Metrics:")
        print("----------------------")
        for metric, value in redability_metrics.items():
            print(f"{metric}: {value}")


    