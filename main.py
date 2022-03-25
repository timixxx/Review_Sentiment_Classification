import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, roc_auc_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import nltk
import pickle

#reading reviews:
print('Loading files...')

def load_txt_data_pos(path_train_pos, path_test_pos):
     tr_files_pos = os.listdir(path_train_pos)
     tr_files_pos = [open(path_train_pos+txt, encoding="utf8").read() for txt in tr_files_pos]
     ts_files_pos = os.listdir(path_test_pos)
     ts_files_pos = [open(path_test_pos+txt, encoding="utf8").read() for txt in ts_files_pos]
     return ts_files_pos+tr_files_pos

def load_txt_data_neg(path_train_neg, path_test_neg):
     tr_files_neg = os.listdir(path_train_neg)
     tr_files_neg = [open(path_train_neg+txt, encoding="utf8").read() for txt in tr_files_neg]
     ts_files_neg = os.listdir(path_test_neg)
     ts_files_neg = [open(path_test_neg+txt, encoding="utf8").read() for txt in ts_files_neg]
     return ts_files_neg+tr_files_neg


#converting to a dataframe:
print('Converting to a dataframe...')

data = []
for row in load_txt_data_pos('source_data/train/pos/', 'source_data/test/pos/'):
     data.append((row, 'pos'))

for row in load_txt_data_neg('source_data/train/neg/', 'source_data/test/neg/'):
     data.append((row, 'neg'))

index = [i for i in range(len(data))]

df = pd.DataFrame(data=data, index=index, columns=['Text', 'Sentiment'])


#removing duplicates:
df = df.drop_duplicates()
#print(df_train.groupby('Sentiment').describe())


#saving dataFrame as csv
#df.to_csv('Movie_Data.csv')


#vectorizing text:
print('Vectorizing...')

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


#Setting a model:
#(need to finish pickle loading of existing model)
#def load_model(path):
#     model_t = open(path, 'rb')
#     model = pickle.load(model_t)
#     model_t.close()
#     return model
#model = load_model('movieSent_LogRegrModel.pkl')

print('Fitting model...')

model = LogisticRegression(solver='liblinear')
#model = BernoulliNB()
#model = MultinomialNB()
#model = KNeighborsClassifier(algorithm='brute', n_jobs=1)
model.fit(x_train, y_train)


#Accuracy tests:
def accuracy_test(model, x_test, y_test):
     probability = model.predict_proba(x_test)
     predicts = model.predict(x_test)

     print('Accuracy tests:')
     print("ROC AUC Score:", roc_auc_score(y_test, probability[:, 1]))
     print("Mean Accuracy of LR Score:", model.score(x_test,y_test))
     print("F1 Score:", f1_score(y_test, predicts, pos_label="pos"))
     print("Accuracy Score:", accuracy_score(y_test, predicts))
     print(classification_report(y_test, predicts, target_names=['Positive', 'Negative']))
     plot_confusion_matrix(model, x_test, y_test, display_labels=['Negative', 'Positive'], cmap='plasma', normalize='true')
     plt.show()

accuracy_test(model, x_test, y_test)


#Testing with real (input) reviews:
def input_test(model):
     print('Enter your review: ')
     text = str(input())
     pred_stars = int(model.predict_proba(vectorizer.transform([text]))[0][1]*10)
     print('Result: \nYour review probably has "', int(pred_stars),'" out of 10 stars')
     if pred_stars >= 7:
          print('Review Status: Positive')
     elif pred_stars <= 4:
          print('Review Status: Negative')
     else:
         print('Review Status: Neutral')

input_test(model)


#saving a model
#pickle.dump(model, open('movieSent_LogRegrModel.pkl', 'wb'))
#pickle.dump(vectorizer.vocabulary_, open('movieSent_vocabulary.pkl', 'wb'))