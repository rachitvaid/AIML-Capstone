import warnings
warnings.filterwarnings('ignore')
import streamlit as st 
import re, string, pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from textblob import Word
from collections import OrderedDict
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use("Agg")
nltk.download('stopwords')
nltk.download('wordnet')
st.set_option('deprecation.showPyplotGlobalUse', False)
    

def load_pickle_models():
    # Load NN model
    loaded_model = tf.keras.models.load_model('Final_Model_Pkl.h5')
    file = open('target_enc_dict.obj','rb')
    # Load label encoder dictionary
    target_enc_dict = pickle.load(file)
    file.close()
    # Load TF-IDF vectorizer
    file2 = open('tfidf_vec.pkl','rb')
    tfidf_vec = pickle.load(file2)
    file2.close()
    return tfidf_vec, target_enc_dict, loaded_model

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text=text.replace(('first name: ').lower(),'firstname')
    text=text.replace(('last name: ').lower(),'lastname')
    text=text.replace(('received from:').lower(),'')
    text=text.replace('email:','')
    text=text.replace('email address:','') 
    index1=text.find('from:')
    index2=text.find('\nsddubject:')
    text=text.replace(text[index1:index2],'')
    index3=text.find('[cid:image')
    index4=text.find(']')
    text=text.replace(text[index3:index4],'')
    text=text.replace('subject:','')
    text=text.replace('received from:','')
    text=text.replace('this message was sent from an unmonitored email address', '')
    text=text.replace('please do not reply to this message', '')
    text=text.replace('monitoring_tool@company.com','MonitoringTool')
    text=text.replace('select the following link to view the disclaimer in an alternate language','')
    text=text.replace('description problem', '') 
    text=text.replace('steps taken far', '')
    text=text.replace('customer job title', '')
    text=text.replace('sales engineer contact', '')
    text=text.replace('description of problem:', '')
    text=text.replace('steps taken so far', '')
    text=text.replace('please do the needful', '')
    text=text.replace('please note that ', '')
    text=text.replace('please find below', '')
    text=text.replace('date and time', '')
    text=text.replace('kindly refer mail', '')
    text=text.replace('name:', '')
    text=text.replace('language:', '')
    text=text.replace('customer number:', '')
    text=text.replace('telephone:', '')
    text=text.replace('summary:', '')
    text=text.replace('sincerely', '')
    text=text.replace('company inc', '')
    text=text.replace('importance:', '')
    text=text.replace('gmail.com', '')
    text=text.replace('company.com', '')
    text=text.replace('microsoftonline.com', '')
    text=text.replace('company.onmicrosoft.com', '')
    text=text.replace('hello', '')
    text=text.replace('hallo', '')
    text=text.replace('hi it team', '')
    text=text.replace('hi team', '')
    text=text.replace('hi', '')
    text=text.replace('best', '')
    text=text.replace('kind', '')
    text=text.replace('regards', '')
    text=text.replace('good morning', '')
    text=text.replace('good afternoon', '')
    text=text.replace('good evening', '')
    text=text.replace('please', '')
    text=text.replace('regards', '')
    text=text.replace('NaN','')
    text=text.replace('can''t','cannot')
    text=text.replace('i''ve','i have')
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\r\n', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\t', '', text)

    text = text.lower()
    return text

def test_data_processing(Short_desc, Desc, Caller):
    
    test_tkt = pd.DataFrame({'Short_desc': [Short_desc], 'Desc': [Desc], 'Caller': [Caller]})
    test_tkt['Full_desc'] = test_tkt['Short_desc'] + ' ' +test_tkt['Desc']
    
    # Clean text from unwanted words punctuation and symbols
    test_tkt['Full_desc'] = test_tkt['Full_desc'].apply(lambda x: clean_text(x))
    
    # Remove caller name from description
    def replace(str1, str2):
        return str1.replace(str2, '')
    test_tkt['Full_desc'] = test_tkt.apply(lambda row: replace(row['Full_desc'], row['Caller']), axis=1)
    test_tkt['Full_desc'] = (test_tkt['Full_desc'].str.split()
                              .apply(lambda x: OrderedDict.fromkeys(x).keys())
                              .str.join(' '))
    
    # Remove stop words
    stop = stopwords.words('english')
    test_tkt['Full_desc'] = test_tkt['Full_desc'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
    
    # Lemmatize
    test_tkt['Full_desc']= test_tkt['Full_desc'].apply(lambda x: " ".join([Word(word).lemmatize() for word in str(x).split()]))
    
    # Tokenization
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    test_tkt['Full_desc'] = test_tkt['Full_desc'].apply(lambda x: tokenizer.tokenize(x))
    def combine_text(list_of_text):
        combined_text = ' '.join(list_of_text)
        return combined_text
    test_tkt['Full_desc'] = test_tkt['Full_desc'].apply(lambda x : combine_text(x))
    
    # Exception when there is insufficient data
    if len((test_tkt.iloc[0]['Full_desc']).split()) < 2:
        raise ValueError("Insufficient info")
    return test_tkt
    
def test_vectorizer(test_tkt, tfidf_vec):
    tckt_tfidf_en = tfidf_vec.transform(test_tkt['Full_desc'])
    # collect the tfid matrix in numpy array
    array_en = tckt_tfidf_en.todense()
    df_tfidf_en = pd.DataFrame(array_en)
    X_test = df_tfidf_en.to_numpy()
    return X_test

def model_prediction(X_test, nn_model, target_enc_dict):
    y_pred_d = nn_model.predict(X_test)
    y_pred_d = np.argmax(y_pred_d, axis=1)
    y_pred_d = pd.DataFrame({'Assignment_group': [y_pred_d]})
    y_pred_d['Assignment_group'] = y_pred_d['Assignment_group'].astype(int)
    y_pred_d['Assignment_group']= y_pred_d['Assignment_group'].map(target_enc_dict)
    Pred_Group = y_pred_d.iloc[0]['Assignment_group']
    return Pred_Group

def main():
    """Ticket Classifier"""
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h1 style="color:white;text-align:center;">IT Ticket Classifier App </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    st.info("Predicting assignment group based on ticket description test")
    Short_Description = st.text_input("Short Description","Type Here...")
    Description = st.text_area("Full Description","Type Here...")
    Caller = st.text_input("Caller Full Name","Type Here...")
    tfidf_vec, target_enc_dict, nn_model= load_pickle_models()
    col1, col2 = st.beta_columns(2)
    if col1.button("Classify"):
        try:
            processed_text = test_data_processing(Short_Description, Description, Caller)
            X_test = test_vectorizer(processed_text, tfidf_vec)
            final_result = model_prediction(X_test, nn_model, target_enc_dict)
            st.success("Predicted Ticket Assignment Group - {}".format(final_result))
            st.sidebar.subheader("About")
            st.sidebar.text("1. Non-English tickets are not supported. \n2. Tickets from following Group clubbed \ninto Miscellaneous.\nThey are :- \n* 71 * 54 * 48 * 69 * 57 * 72 * 63 * 49\n* 56 * 68 * 38 * 58 * 66 * 46 * 42 * 59\n* 43 * 52 * 55 * 51 * 30 * 65 * 62 * 53\n* 36 * 50 * 44 * 37 * 27 * 39 * 23 * 33\n* 47 * 1 * 21 * 11 * 22 * 31 * 28 * 20\n* 45 * 15 * 41")
        except ValueError:
            st.error('Insufficient description information')
    if col2.button("NLP Word Cloud"):
        try:
            p_text = test_data_processing(Short_Description, Description, Caller)
            c_text = p_text.iloc[0]['Full_desc']
            wordcloud = WordCloud().generate(c_text)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
        except ValueError:
            st.error('Insufficient description information')
        
        
if __name__ == '__main__':
    main()