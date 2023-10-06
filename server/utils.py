from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from stopwordsiso import stopwords
import pandas as pd, re

def remove_emojis(data):
    emoj = re.compile("["
                      u"\U00002700-\U000027BF"  # Dingbats
                      u"\U0001F600-\U0001F64F"  # Emoticons
                      u"\U00002600-\U000026FF"  # Miscellaneous Symbols
                      u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
                      u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                      u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                      u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def remove_unicode_chars(data):
    pattern = re.compile(u"[\u200c-\u200f\u202a-\u202f\u2066-\u2069]")
    return pattern.sub("", data)


def Preprocessing_for_Marathi_Language(marathi_text):
    # Remove Emojis
    marathi_text = remove_emojis(marathi_text)
    
    # Removing Punctuations
    punctuations = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    punctuation_removed_text = marathi_text
    for ele in marathi_text:
        if ele in punctuations:
            punctuation_removed_text = punctuation_removed_text.replace(ele, " ")

    # Tokenization
    tokenized_text = punctuation_removed_text.split(" ")

    # Remove Spaces
    tokenized_text = list(filter(("").__ne__, tokenized_text))

    # Filter only marathi words
    final_words = list()
    for word in tokenized_text:
        word = remove_unicode_chars(word)
        if len(word) == 0:
            continue
        if any(char.isdigit() for char in word):
            continue
        if not ('a' <= word[0] <= 'z' or 'A' <= word[0] <= 'Z' or word[0].isdigit() or '\n' in word):
            final_words.append(word)

    # Removing Stopwords
    stopwords_removed_text = list()
    stopwords_collection = stopwords('mr')
    for i in final_words:
        if i not in stopwords_collection:
            stopwords_removed_text.append(i)

    # Remove Spaces
    final_list = list()
    for token in stopwords_removed_text:
        if token != "":
            final_list.append(token)
    return " ".join(final_list)

def convert(data):
    df = pd.read_csv("../dataset/marathi.csv")
    x = df.iloc[:, 0]

    train=list()
    for i in range(len(x)):
        train.append(Preprocessing_for_Marathi_Language(x[i]))
    x=pd.Series(train)
    X_train, X_test= train_test_split(x, random_state=3)

    vectorizer = TfidfVectorizer(decode_error="ignore")
    vectorizer.fit_transform(X_train)
    vectorizer.transform(X_test)
    
    return vectorizer.transform(data)