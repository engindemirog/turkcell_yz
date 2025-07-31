#1 - Metin temizleme
import re

text = "Merhaba! Ben doğal dil işleme öğreniyorum ve NLP harika bir alan :)"

text = text.lower()

text = re.sub(r'[^\w\s]','',text)

print(text)

#Tokenization

from nltk.tokenize import word_tokenize

tokens = word_tokenize(text,language="turkish")

print(tokens)

#stopword

from nltk.corpus import stopwords

stopwords = set(stopwords.words("turkish"))

print(stopwords)

filtered_tokens = [word for word in tokens if word not in stopwords]

print(filtered_tokens)

#stemming /lemmatization

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running","flies","better","easily","learning"]

print("Stemming : ", [stemmer.stem(w) for w in words])
print("Lemmatization : ", [lemmatizer.lemmatize(w) for w in words])

#Vector
from sklearn.feature_extraction.text import TfidfVectorizer

data = [
    "Natural language is fun",
    "Learning NLP opens many doors",
    "Studying NLP makes you feel happy"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)

print(vectorizer.get_feature_names_out())
print(X.toarray())