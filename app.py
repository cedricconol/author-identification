import streamlit as st

st.sidebar.header('Options')

vectorizer = st.sidebar.selectbox('Select feature extraction method:',
 ('Bag of Words', 'TF-IDF', 'Doc2vec'), index=2)

st.title('Author Identification')

st.markdown("A web-app from the paper, _[Effectiveness of Doc2Vec in Authorship Identification Tasks](https://cedricconol.github.io/doc2vec/)_, "
"by Cedric Conol and Rosely Peña.")

st.markdown('This app identifies the most probable author of a sentence or phrase'
        ' from a given list of authors. The logistic regression model used to predict the author'
        ' was trained using the data from the [`Spooky Author Identification`]'
        '(https://www.kaggle.com/c/spooky-author-identification/data) competition in '
        'Kaggle. ')

showinfo = st.checkbox('Show dataset information')

if showinfo:
    st.markdown(">_\"The competition dataset contains text from works of fiction written by spooky authors of the public domain: Edgar Allan Poe, HP Lovecraft and Mary Shelley. The data was prepared by chunking larger texts into sentences using CoreNLP's MaxEnt sentence tokenizer, so you may notice the odd non-sentence here and there. Your objective is to accurately identify the author of the sentences in the test set.\"_")

st.subheader('Give it a try!')

sent = st.text_input("Write a sentence in the textbox below and press Enter."
" You may use a random sentence from the test set using the sidebar.",
 'Still, as I urged our leaving Ireland with such inquietude and impatience, my father thought it best to yield.', key="textinput")

import pandas as pd
from random import randint

test = pd.read_csv('data/test.csv', usecols=['text'])
st.sidebar.header("")
st.sidebar.subheader("Random Sentence")
st.sidebar.text('Generate random sentence from test set.')

if st.sidebar.button('Go!'):
    random_int = randint(0, len(test))
    random_sent = test.loc[random_int]['text']
    st.sidebar.text(random_sent)

import pickle

# doc2vec
from gensim.models.doc2vec import TaggedDocument
import nltk

def d2v(sent):
    td = TaggedDocument(words=nltk.word_tokenize(sent), tags=[0])

    with open('pickles/dbow.pkl', 'rb') as f:
        dbow = pickle.load(f)

    vectorize = dbow.infer_vector(td.words, steps=20)

    with open('pickles/logreg.pkl', 'rb') as f:
        model = pickle.load(f)

    pred = model.predict(vectorize.reshape(1,-1))[0]
    proba = max(model.predict_proba(vectorize.reshape(1,-1))[0])*100
    return pred, proba

# tfidf
def tfidf(sent):
    with open('pickles/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    vectorize = tfidf.transform([sent])

    with open('pickles/logreg_tfidf.pkl', 'rb') as f:
        model = pickle.load(f)
    
    pred = model.predict(vectorize.reshape(1,-1))[0]
    proba = max(model.predict_proba(vectorize.reshape(1,-1))[0])*100
    return pred, proba

# bow
def bow(sent):
    with open('pickles/bow.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    vectorize = tfidf.transform([sent])

    with open('pickles/logreg_bow.pkl', 'rb') as f:
        model = pickle.load(f)
    
    pred = model.predict(vectorize.reshape(1,-1))[0]
    proba = max(model.predict_proba(vectorize.reshape(1,-1))[0])*100

    return pred, proba

def fullname(pred_author):
    if pred_author==0:
        return 'Edgar Allan Poe'
    elif pred_author==1:
        return 'HP Lovecraft'
    else:
        return 'Mary Shelley'

def showimage(pred_author):
    if pred_author==0:
        return 'images/eap.jpg'
    elif pred_author==1:
        return 'images/hpl.jpg'
    else:
        return 'images/mws.jpg'

# predict
if vectorizer=="Bag of Words":
    pred, proba = bow(sent)
elif vectorizer=="TF-IDF":
    pred, proba = tfidf(sent)
else:
    pred, proba = d2v(sent)

from PIL import Image
image = Image.open(showimage(pred))

st.image(image, width=image.size[1]//10)
st.subheader('Prediction:')
st.write("I am", str(proba)[:5]+"%", " sure it's written by ", fullname(pred), ".")

