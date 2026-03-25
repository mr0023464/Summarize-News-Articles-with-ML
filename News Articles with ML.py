import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')


text = """Python is a very popular general-purpose programming language which was created 
by Guido van Rossum, and released in 1991. It is very popular for web development 
and you can build almost anything like mobile apps, web apps, tools, data analytics,
machine learning etc. It is designed to be simple and easy like english language. 
It's is highly productive and efficient making it a very popular language."""


sentences = sent_tokenize(text)


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sentences)


scores = np.sum(X.toarray(), axis=1)


top_n = 2
top_sentences = scores.argsort()[-top_n:]


print("Summary:\n")

for i in top_sentences:
    print(sentences[i])