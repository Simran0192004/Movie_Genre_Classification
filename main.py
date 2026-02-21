#------------------------------------------------Importing libraries---------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

#---------------------------------------------Setting up the dataset----------------------------------------------
data = []

with open(r'Genre Classification Dataset\train_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(':::')

        if len(parts) == 4:
            movie_id = parts[0].strip()
            name = parts[1].strip()
            genre = parts[2].strip()
            plot = parts[3].strip()

            data.append([genre, plot])

df = pd.DataFrame(data, columns=['genre', 'plot'])

print("Total rows:", len(df))

#-------------------------------------------Text Preprocessing-----------------------------------------------
#Lowercasing
df['plot']=df['plot'].str.lower()

#Removing the punctuations
punctuations = string.punctuation

def tokenize_text(text):
    return word_tokenize(text)

def remove_punctuation(text):
    new_text=[]
    for word in tokenize_text(text):
        if word not in punctuations:
            new_text.append(word)
    return ' '.join(new_text)

df['plot']=df['plot'].apply(remove_punctuation)

#Removing stopwords
stop_words=stopwords.words('english')

def remove_stopwords(text):
    new_text=[]
    for word in text.split():
        if word not in stop_words:
            new_text.append(word)
    return ' '.join(new_text)

df['plot']=df['plot'].apply(remove_stopwords)


#Lemmatization
word_lem=WordNetLemmatizer()

def lemmatize_text(text):
    new_text=[]
    for word in text.split():
        new_text.append(word_lem.lemmatize(word))
    return ' '.join(new_text)

df['plot']=df['plot'].apply(lemmatize_text)

counts = df['genre'].value_counts()
valid_genres = counts[counts >= 1500].index
df = df[df['genre'].isin(valid_genres)]


#Splitting the dataset into features and target variable
X=df['plot'].values
Y=df['genre'].values

#Applying the Label Encoder to convert the target variables into numerical values or labels
lb=LabelEncoder()
Y=lb.fit_transform(Y)


#Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)

#Applying the TF-IDF vectorizer to convert the text data into numerical values
tfidf=TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9
)
X_train=tfidf.fit_transform(X_train)
X_test=tfidf.transform(X_test)


#Applying different machine learning models to the dataset and evaluating their performance
models={
    'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'naive_bayes': MultinomialNB(),
    'logistic_regression': LogisticRegression(max_iter=2000, C=1.5, n_jobs=-1),
    'linear_svc': LinearSVC(class_weight='balanced')

}

results={}

for model_name in models:
    model=models[model_name]
    model.fit(X_train, Y_train)
    print(f"{model_name} trained successfully")
    Y_pred=model.predict(X_test)
    results[model_name] = {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, average='weighted'),
        "recall": recall_score(Y_test, Y_pred, average='weighted'),
        "macro_f1": f1_score(Y_test, Y_pred, average='macro')
    }

results_df = pd.DataFrame(results).T
print(results_df.sort_values("macro_f1", ascending=False))
print('\n')
best_model_name = results_df['macro_f1'].idxmax()
print("Best model:", best_model_name)
print('\n')
cm = confusion_matrix(Y_test, Y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=models[best_model_name].classes_
)

disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()