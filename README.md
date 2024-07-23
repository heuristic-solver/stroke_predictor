#Stroke Prediction using TensorFlow and Keras Preprocessing

This project demonstrates how to use TensorFlow and Keras preprocessing layers to build and train a neural network for predicting strokes. The dataset used for this project is the Healthcare Stroke Dataset.
Table of Contents

    Introduction
    Dataset
    Data Preprocessing
    Model Building
    Training the Model
    Usage
    Results
    License

Introduction

This project aims to predict strokes using a neural network built with TensorFlow and Keras. The preprocessing steps leverage Keras preprocessing layers for normalization and categorical encoding, demonstrating a streamlined approach to preparing data for machine learning models.
Dataset

The dataset used is the healthcare-dataset-stroke-data.csv, which includes various features such as age, gender, BMI, and average glucose level. The target variable is stroke, indicating whether the patient has had a stroke.

python

import pandas as pd

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data = data.drop(columns=['smoking_status', 'Residence_type', 'id'])
print(data.head())

Data Preprocessing

The preprocessing involves normalizing numerical features and encoding categorical features using Keras preprocessing layers.
Normalization

We normalize numerical features to ensure they have a mean of 0 and a standard deviation of 1.

python

from tensorflow.keras.layers.experimental import preprocessing

def norm_layer(name, dataset):
    norm = preprocessing.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name])
    norm.adapt(feature_ds)
    return norm

nm_col = train_features['age']
layer = norm_layer('age', train_ds)
print(layer(nm_col))

Categorical Encoding

We encode categorical features using Keras preprocessing layers to convert string values to integer indices and then to one-hot encoded vectors.

python

def category_encoder(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)
    
    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    
    cat_layer = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())
    
    return lambda feature: cat_layer(index(feature))

type_col = train_features['gender']
layer = category_encoder('gender', train_ds, 'string')
print(layer(type_col))

Model Building

We build a neural network using TensorFlow Keras. The model includes an input layer for each feature, followed by a concatenation of the encoded features, and finally, dense and dropout layers.

python

import tensorflow as tf

num_cols = ['age', 'avg_glucose_level', 'bmi']
cat_cols = ['gender', 'ever_married', 'work_type']

feature = []
encoded_feature = []

for header in num_cols:
    num_column = tf.keras.Input(shape=(1,), name=header)
    normalizer = norm_layer(header, train_ds)
    enc_col = normalizer(num_column)
    encoded_feature.append(enc_col)
    feature.append(num_column)

for header in cat_cols:
    str_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoded_str = category_encoder(header, train_ds, 'string', max_tokens=5)
    enc_vals = encoded_str(str_col)
    feature.append(str_col)
    encoded_feature.append(enc_vals)

all_features = tf.keras.layers.concatenate(encoded_feature)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(feature, output)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

Training the Model

We split the data into training, validation, and test sets and train the model for 10 epochs.

python

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

batch_size = 256

train_ds = make_dataset(train, batch_size)
test_ds = make_dataset(test, batch_size)
val_ds = make_dataset(val, batch_size)

model.fit(train_ds, epochs=10, validation_data=val_ds)

Usage

To use this project, clone the repository and run the provided code. Ensure you have the necessary dependencies installed, including TensorFlow and pandas.

sh

git clone https://github.com/your-username/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt

Results

The model's performance will be displayed after training. You can further evaluate it on the test set using the evaluate method.
License

This project is licensed under the MIT License - see the LICENSE file for details.
