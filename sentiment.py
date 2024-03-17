#important libraries
import pandas as pandas
import numpy as np
import re
import string

#methods for text processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#machine learning libraries

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes.MultinomialNB import MultinomialNB
from sklearn.linear_model.LoginticRegression import  LoginticRegression
from sklearn.svm.SVC import SVC

import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))


#to load the data set

def load_dataset(filepath, cols):
    """
    reads the CSV file to return 
    a dataframe with specified column names
    """
    df=pd.read_csv(filepath, encoding='latin-1')
    df.colmns=cols
    return df

def delete_redundant_cols(df, cols):
    for col in cols:
        del df[col]
    return df    


#preprocessing tasks
# 1.casing
# 2.noise Removal
# 3.Tokenization
# 4.stopword Removal
# 5.text Normalization(stemming and lemmatization)


def preprocess_tweet_text(tweet):
    #convert all text to lower case
    tweet=tweet.lower()

    #remove any urls
    tweet=re.sub(r"http\S+|www\s+|https\s+","", tweet,flags=re.MULTILINE)

    #remove punctuation

    tweet=tweet.translate(str.maketrans("","",string.punctuation))
    
    #
    tweet=re.sub(r'\@\w+|\#', "",tweet)

    tweet_tokens=word_tokenize(tweet)
    filetred_words={word for word in tweet_tokens if word not in stop_words}

    ps=PorterStemmer()
    stemmed_words={ps.stem(w) for w in filetred_words}

    lemmatizer=WordnetLemmatizer()
    lemma_words={lemmatizer.lemmatize(w,pos='a') for w in stemmed_words}

    return " ".join(lemma_words)


preprocess_tweet_text("ji there , how are you preparing for your exams?")  



def get_feature_vector(train_fit):
    vector=TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

