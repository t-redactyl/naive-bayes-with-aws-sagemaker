import os
import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    train_data_dir = os.environ['SM_CHANNEL_TRAIN']
    model_output = os.environ['SM_MODEL_DIR']
    ouptut_dir = os.environ['SM_OUTPUT_DATA_DIR']
    
    # Read in training data
    train_data = os.path.join(train_data_dir, 'train_titles.csv')
    full_data = os.path.join(train_data_dir, 'full_titles.csv')
    if not os.path.exists(train_data):
        raise ValueError(f'File {train_data} is not there')
        
    train = pd.read_csv(train_data, header=0)
    full = pd.read_csv(full_data, header=0)
    
    # Apply transformations to features and target
    # Features: convert into sparse matrix with Tf-Idf weights
    train_x = train["product_title"].apply(lambda x: np.str_(x))
    
    tfidf_vect = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
    tfidf_vect.fit(full["product_title"].apply(lambda x: np.str_(x)))
    train_x_tfidf = tfidf_vect.transform(train_x)
        
    # Target: Encode as numeric
    train_y = train["product_category"]
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    
    nb_model = MultinomialNB()
    nb_model = nb_model.fit(train_x_tfidf, train_y)
    
    joblib.dump(nb_model, os.path.join(model_output, "model.joblib"))
    
    
def model_fn(model_output):
    
    nb_model = joblib.load(os.path.join(model_output, "model.joblib"))
    return nb_model