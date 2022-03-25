from django.shortcuts import render
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv('Movie_Data.csv')

train_corpus, test_corpus, y_train, y_test = train_test_split(
    df.Text,
    df.Sentiment,
    test_size=0.2,
    shuffle='true',
)

stop_words = nltk.corpus.stopwords.words('english')
vectorizer =TfidfVectorizer(min_df=22, ngram_range=(1,2), stop_words=stop_words)
x_train = vectorizer.fit_transform(train_corpus)
x_test = vectorizer.transform(test_corpus)

model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

#model = pickle.load(open('Movie_SentimentCls/movieSent_LogRegrModel.pkl'))

def index(request):
    context = {'a':'null'}
    return render(request, 'index.html', context)


def predictRev(request):
    if request.method == 'POST':
        temp = {}
        temp['review'] = request.POST.get('reviewVal')
    pred_stars = int(model.predict_proba(vectorizer.transform([request.POST.get('reviewVal')]))[0][1] * 10)
    if pred_stars <= 4:
        status = 'Negative'
    elif pred_stars >= 7:
        status = 'Positive'
    else:
        status = 'Neutral'

    context = {'Predicted_Score':pred_stars, 'Status': status}
    return render(request, 'index.html', context)