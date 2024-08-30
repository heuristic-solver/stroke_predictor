import numpy as np 
import tensorflow as tf 
import pandas as pd 
from tensorflow.keras.layers.experimental import preprocessing 
from sklearn.model_selection import train_test_split 


data = pd.read_csv('healthcare-dataset-stroke-data.csv') 


data = data.drop(columns=['smoking_status','Residence_type','id'])
print(data.head())

def make_dataset(data,shuffle = True, batch_size =32): #data preprocessing
    dataframe = data.copy()
    label = dataframe.pop('stroke')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),label))
    
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    
    return ds 

batch_size = 5 
train_ds = make_dataset(data,batch_size)
print(train_ds)


[(train_features, label_batch)] = train_ds.take(1)

def norm_layer(name,dataset):
    norm = preprocessing.Normalization(axis = None)
    feature_ds = dataset.map(lambda x,y:x[name])
    
    norm.adapt(feature_ds)
    return norm

nm_col = train_features['age']
layer = norm_layer('age',train_ds)
print(layer(nm_col))


def category_encoder(name,dataset,dtype,max_tokens=None):
    
    if dtype=='string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)
    
    feature_ds = dataset.map(lambda x,y:x[name])
    index.adapt(feature_ds)
    
    cat_layer = preprocessing.CategoryEncoding(num_tokens = index.vocabulary_size())
    
    return lambda feature:cat_layer(index(feature))

type_col = train_features['gender']
layer = category_encoder('gender',train_ds,'string')
print(layer(type_col))

train,test = train_test_split(data,test_size=0.2)
train,val = train_test_split(train,test_size=0.2)

batch_size = 256

train_ds = make_dataset(train,batch_size)
test_ds = make_dataset(test,batch_size)
val_ds = make_dataset(val,batch_size)

num_cols = ['age','avg_glucose_level','bmi']


feature = []
encoded_feature = []

for header in num_cols:
    num_column = tf.keras.Input(shape=(1,),name=header)
    normalizer = norm_layer(header,train_ds)
    
    enc_col = normalizer(num_column)
    encoded_feature.append(enc_col)
    feature.append(num_column)
    



cat_cols = ['gender','ever_married','work_type']
for header in cat_cols:
    str_col = tf.keras.Input(shape=(1,),name=header,dtype='string')
    encoded_str = category_encoder(header,train_ds,'string',max_tokens=5)
    enc_vals = encoded_str(str_col)
    
    feature.append(str_col)
    encoded_feature.append(enc_vals)
    
all_features = tf.keras.layers.concatenate(encoded_feature)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(feature, output)
# TODO
# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])


model.fit(train_ds,epochs=10,validation_data=val_ds)
    


