import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import re
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier



ps=PorterStemmer()

data=pd.read_csv("IMDB Dataset.csv")
#print(data)


def transform(text):
    pattern = r'<br />'
    l=re.findall(pattern,text)
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]

    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation and i not  in l:
            y.append(i)

    text=y[:]

    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


data['review']=data['review'].apply(transform)



#checking for minority class
#print(data.value_counts('sentiment'))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load example dataset (you can replace this with your own dataset)

le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])



cv=CountVectorizer()
X=cv.fit_transform(data['review']).toarray()

y=data['sentiment'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)




rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = rf_classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
print(confusion_matrix(y_test,predictions))
import pickle
pickle.dump(cv,open('vectorizer.pkl','wb'))
pickle.dump(rf_classifier,open('rfmodel.pkl','wb'))
