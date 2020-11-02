import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re  # to simplify the reviews
import nltk  # that allow us to download and use stop words (words which are not helpful if they are negative or positive)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # allow us to stemming which is (only we are taking the root of the words)
from sklearn.feature_extraction.text import CountVectorizer


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)  # delimiter because we want to open tsv text, quoting parameter to avoiding " " "

# cleaning text
nltk.download('stopwords')
corpus = []  # we are providing cleaning to all words in reviews
for i in range(1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # to remove all punctuations by replacing everything what is not a letter into space
    review = review.lower()  # it return only small letters in all reviews
    review = review.split()  # split root form other part of the word like ed
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')  # exluding word 'not' from stopwords
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]  # to get rid of unnecessary words
    review = ' '.join(review)  # to join the review
    corpus.append(review)

print(corpus)
# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)  # parameter is max amount words in vector
X = cv.fit_transform(corpus).toarray()  # it will take all the words into the columns, to array because naive bayes model
y = dataset.iloc[:, -1].values
print(len(X[0]))

# splitting the dataset into the Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ass = accuracy_score(y_test, y_pred)
print(ass)
