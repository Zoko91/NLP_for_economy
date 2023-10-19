import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from contractions import contractions_dict

# Download necessary ntlk packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences


def tokenize_words(sentence):
    words = word_tokenize(sentence)
    return words


def remove_special_characters(text):
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def expand_contractions(text):
    # Expand contractions (e.g., "don't" to "do not")
    words = text.split()
    expanded_words = [contractions_dict.get(word, word) for word in words]
    return ' '.join(expanded_words)


def lowercase_text(text):
    # Lowercase all words, might be interesting to keep the Names ? Tagging necessary for that ?
    return text.lower()


def remove_stopwords(words):
    # Remove stopwords (e.g., "the", "a", "an", "in")
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


def lemmatize_words(words):
    # Lemmatizing: Transforming words to their base form (e.g., "dogs" to "dog", "running" to "run")
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    tagged_words = pos_tag(words)  # Perform Part-of-Speech tagging
    for word, tag in tagged_words:
        if tag.startswith('V'):  # Verb
            word = lemmatizer.lemmatize(word, 'v')  # Lemmatize verb to its base form
        lemmatized_words.append(word)
    return lemmatized_words


def clean_text(text):
    # This function can also be done in paragraphs but the result might vary depending
    # on the indentation of the document
    sentences = tokenize_sentences(text)

    cleaned_sentences = []
    for sentence in sentences:
        # Remove special characters, expand contractions, tokenize words, lowercase words,
        sentence = remove_special_characters(sentence)
        sentence = expand_contractions(sentence)
        words = tokenize_words(sentence)
        words = [word.lower() for word in words]
        words = remove_stopwords(words)
        words = lemmatize_words(words)
        cleaned_sentences.append(' '.join(words))

    return ' '.join(cleaned_sentences), cleaned_sentences