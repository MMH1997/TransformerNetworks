#!/usr/bin/env python
# coding: utf-8

# ### Import libraries. 

# In[1]:


get_ipython().system('pip install -U numpy==1.18.5')
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from tensorflow import keras
import sklearn
import keras
import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import sklearn.model_selection
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed


# ### Data preprocessing

# Importing dataset. 

# In[2]:


data = pd.read_csv('DatosPreprocesadosH.csv')
data = data.loc[:, data.columns != 'Unnamed: 0']
data['fecha'] = pd.to_datetime(data['fecha'])
data.head()


# Selecting variables. 

# In[4]:


dataset = data.iloc[:, 6:20]


# In[5]:


dataset


# Transformation of Ozone concentration in a categorical variable.

# In[9]:


for i in range(dataset.shape[0]):
    if (dataset.iloc[i,13] < 60):
        dataset.iloc[i,13] = 0
    elif (60 < dataset.iloc[i,13] < 120):
        dataset.iloc[i,13] = 1
    else:
        dataset.iloc[i,13] = 2


# Printing the number of observations in each category 

# In[10]:


print(list(dataset['O3']).count(0))
print(list(dataset['O3']).count(1))
print(list(dataset['O3']).count(2))
# print(list(dataset['O3']).count(3))


# Function which converts data in Time Series format to data in Supervised Learning format

# In[11]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Applying an econder to categorical variables

# In[12]:


values = dataset.values
encoder = LabelEncoder()
values[:,11] = encoder.fit_transform(values[:,11])
values[:,12] = encoder.fit_transform(values[:,12])
values[:,13] = encoder.fit_transform(values[:,13])


# Scaling all variables

# In[14]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# Selecting desired time in advance (24 hours in this case) and the corresponding variables. 

# In[15]:


reformed = series_to_supervised(scaled,24)
reformed=reformed.iloc[:,0:56]
reformed['o3'] = dataset["O3"]
reformed.head()


# Splitting into train/test 

# In[16]:


V = reformed.values
train = V[:30000, :]
test = V[30000:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
train_X = np.asarray(train_X).astype('float32')
test_X = np.asarray(test_X).astype('float32')
labelencoder = LabelEncoder()
train_y = labelencoder.fit_transform(train_y)
test_y = labelencoder.fit_transform(test_y)


# ### Deep Transformer Network

# Developing the transformer model

# In[17]:


n_classes = len(np.unique(train_y))
input_shape = train_X.shape[1:]
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-4)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


# In[18]:


input_shape


# Fitting the hyperparameters of the model

# In[19]:


model = build_model(
    input_shape,
    head_size=2,
    num_heads=2,
    ff_dim=1,
    num_transformer_blocks=1,
    mlp_units=[500],
    mlp_dropout=0.2,
    dropout=0.15,
)


# Visualising the neural network using Netron 

# In[23]:


pip install netron
import netron
model.save('model.h5')
netron.start('model.h5')


# Compilation and summary of the model

# In[ ]:


tf.random.set_seed(232323)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
)
model.summary()


# Setting the callbacks

# In[ ]:


callbacks = [keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)]


# Testing the hyperparameters values.

# In[ ]:


a = list()
b = list()
for k in range(3,20,3):
    for j in range(3,20,3):
        for f in range(5,30,5):
            model = build_model(
            input_shape,
            head_size=k,
            num_heads=j,
            ff_dim=f,
            num_transformer_blocks=1,
            mlp_units=[5000],
            mlp_dropout=0.2,
            dropout=0.15,
            )
            tf.random.set_seed(232323)
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer="adam",
                metrics=keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            )
            model.summary()
            tf.random.set_seed(3444)
            model.fit(
                train_X,
                train_y,
                validation_split = 0.02,
                epochs=20,
                batch_size=8,
                callbacks=callbacks,
            )

            pred = model.predict(test_X)
            predi = list()
            for i in range(pred.shape[0]):
                max_value = max(list(pred[i]))
                max_index = list(pred[i]).index(max_value)
                predi.append(max_index)
            cm = confusion_matrix(test_y.reshape(test_X.shape[0]),predi)


            acc=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predi)
            bac=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predi)
            a.append(acc)
            b.append(bac)


# In[ ]:


a = list()
b = list()
for _ in range(20):
    tf.random.set_seed(3444)
    model.fit(
    train_X,
    train_y,
    validation_split = 0.1,
    epochs=15,
    batch_size=8,
    callbacks=callbacks,
    )
    pred = model.predict(test_X)
    predi = list()
    for i in range(pred.shape[0]):
        max_value = max(list(pred[i]))
        max_index = list(pred[i]).index(max_value)
        predi.append(max_index)
    cm = confusion_matrix(test_y.reshape(test_X.shape[0]),predi)
    cm
    acc=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predi)
    bac=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predi)
    a.append(acc)
    b.append(bac)


# Fitting the model

# In[ ]:


model.fit(
    train_X,
    train_y,
    validation_split = 0.02,
    epochs=35,
    batch_size=8,
    callbacks=callbacks,
)
pred = model.predict(test_X)
predi = list()
for i in range(pred.shape[0]):
    max_value = max(list(pred[i]))
    max_index = list(pred[i]).index(max_value)
    predi.append(max_index)
cm = confusion_matrix(test_y.reshape(test_X.shape[0]),predi)
cm
acc=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predi)
bac=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predi)
print(bac), print(acc)


# Predicting in testing set and printing the confusion matrix.

# In[ ]:


pred = model.predict(test_X)
predi = list()
for i in range(pred.shape[0]):
    max_value = max(list(pred[i]))
    max_index = list(pred[i]).index(max_value)
    predi.append(max_index)
cm = confusion_matrix(test_y.reshape(test_X.shape[0]),predi)
cm


# Evaluation of the model

# In[ ]:


acc=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predi)
bac=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predi)
print(bac), print(acc)


# ### MultiLayerPerceptron

# Reshaping the data

# In[ ]:


train_X1 = train_X.reshape(train_X.shape[0],train_X.shape[2])
train_X1 = np.asarray(train_X1).astype('float32')
test_X1 = test_X.reshape(test_X.shape[0],test_X.shape[2])
test_X1 = np.asarray(test_X1).astype('float32')


# Development, compilation and summary of the MLP network

# In[ ]:


modelmlp = Sequential()
modelmlp.add(Dense(64, input_dim=test_X.shape[2], activation= 'relu'))
modelmlp.add(Dense(64, activation='relu'))
modelmlp.add(Dense(16, activation='relu'))
modelmlp.add(Dense(8, activation='relu'))
modelmlp.add(Dense(8, activation='relu'))
# modelmlp.add(Dense(16, activation='relu'))
modelmlp.add(Dense(3, activation='softmax'))
modelmlp.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=["sparse_categorical_accuracy"])
print(modelmlp.summary())


# Fitting the model

# In[ ]:


tf.random.set_seed(11111)
modelmlp.fit(train_X1, train_y, batch_size=16, epochs=19, validation_split=0.1, verbose=1)


# Predicting in testing set and printing the confusion matrix

# In[ ]:


predmlp = modelmlp.predict(test_X.reshape(test_X.shape[0],test_X.shape[2]))
predimlp = list()
for i in range(predmlp.shape[0]):
    max_value = max(list(predmlp[i]))
    max_index = list(predmlp[i]).index(max_value)
    predimlp.append(max_index)
cmmlp = confusion_matrix(test_y.reshape(test_X.shape[0]),predimlp)
cmmlp


# Evaluation of the model

# In[ ]:


accmlp=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predimlp)
bacmlp=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predimlp)
print(bacmlp), print(accmlp)


# Evaluation of the model in 20 replications

# In[ ]:


train_X1 = train_X.reshape(train_X.shape[0],train_X.shape[2])
train_X1 = np.asarray(train_X1).astype('float32')
test_X1 = test_X.reshape(test_X.shape[0],test_X.shape[2])
test_X1 = np.asarray(test_X1).astype('float32')
a = list()
b = list()
for _ in range(20):
    modelmlp = Sequential()
    modelmlp.add(Dense(512, input_dim=test_X.shape[2], activation= 'relu'))
    modelmlp.add(Dense(256, activation='relu'))
    modelmlp.add(Dense(128, activation='relu'))
    modelmlp.add(Dense(32, activation='relu'))
    modelmlp.add(Dense(32, activation='relu'))
    # modelmlp.add(Dense(16, activation='relu'))
    modelmlp.add(Dense(3, activation='softmax'))
    modelmlp.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=["sparse_categorical_accuracy"])
    print(modelmlp.summary())
    modelmlp.fit(train_X1, train_y, batch_size=16, epochs=20, validation_split=0.1, verbose=1)
    predmlp = modelmlp.predict(test_X.reshape(test_X.shape[0],test_X.shape[2]))
    predimlp = list()
    for i in range(predmlp.shape[0]):
        max_value = max(list(predmlp[i]))
        max_index = list(predmlp[i]).index(max_value)
        predimlp.append(max_index)
    cmmlp = confusion_matrix(test_y.reshape(test_X.shape[0]),predimlp)
    accmlp=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predimlp)
    bacmlp=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predimlp)
    a.append(accmlp)
    b.append(bacmlp)


# ### Random Forest

# Development and trainning of the Random Forest model.

# In[ ]:


rf150 = RandomForestClassifier(n_estimators = 250)
rf150.fit(train_X1, train_y)


# Predicting and printing the confusion matrix

# In[ ]:


predrf = rf150.predict(test_X.reshape(test_X.shape[0],test_X.shape[2]))
cmrf = confusion_matrix(test_y.reshape(test_X.shape[0]),predrf)
cmrf


# Evaluation of the model

# In[ ]:


accrf=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predrf)
bacrf=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predrf)
print(bacrf), print(accrf)


# Evaluation of the model in 20 replications

# In[ ]:


a = list()
b = list()
for _ in range(15):
    rf150 = RandomForestClassifier(n_estimators = 250)
    rf150.fit(train_X1, train_y)
    predrf = rf150.predict(test_X.reshape(test_X.shape[0],test_X.shape[2]))
    cmrf = confusion_matrix(test_y.reshape(test_X.shape[0]),predrf)
    accrf=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predrf)
    bacrf=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predrf)
    a.append(accrf)
    b.append(bacrf)


# ### LSTM

# Development and compilation of LSTM network.

# In[ ]:


modelB1008 = Sequential()
modelB1008.add(LSTM(150, input_shape=(1,56)))
modelB1008.add(Dense(15, activation='relu'))
modelB1008.add(Dense(5, activation='relu'))
modelB1008.add(Dense(5, activation='relu'))
modelB1008.add(Dense(3, activation='softmax'))
modelB1008.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=["sparse_categorical_accuracy"])
modelB1008.summary()


# In[ ]:


history50016 = modelB1008.fit(train_X, train_y, epochs=25, batch_size=16, 
                    validation_split=0.2, verbose=1, shuffle=False)


# Predicting in testing set and printing the confusion matrix

# In[ ]:


predlstm = modelB1008.predict(test_X)
predilstm = list()
for i in range(predlstm.shape[0]):
     max_value = max(list(predlstm[i]))
     max_index = list(predlstm[i]).index(max_value)
     predilstm.append(max_index)
cmlstm = confusion_matrix(test_y,predilstm)
cmlstm


# Evaluation of the model

# In[ ]:


accmlp=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predilstm)
bacmlp=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predilstm)
print(bacmlp), print(accmlp)


# ### 

# Evaluation of the model in 20 replications

# In[ ]:


a = list()
b = list()
for _ in range(20):
    modelB1008 = Sequential()
    modelB1008.add(LSTM(12, input_shape=(1,56)))
#     modelB1008.add(Dense(15, activation='relu'))
#     modelB1008.add(Dense(5, activation='relu'))
#     modelB1008.add(Dense(5, activation='relu'))
    modelB1008.add(Dense(3, activation='softmax'))
    modelB1008.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=["sparse_categorical_accuracy"])
    modelB1008.summary()


# In[ ]:


    history50016 = modelB1008.fit(train_X, train_y, epochs=25, batch_size=16, 
                    validation_split=0.2, verbose=1, shuffle=False)
    predlstm = modelB1008.predict(test_X)
    predilstm = list()
    for i in range(predlstm.shape[0]):
         max_value = max(list(predlstm[i]))
         max_index = list(predlstm[i]).index(max_value)
         predilstm.append(max_index)
    cmlstm = confusion_matrix(test_y,predilstm)
    cmlstm
    accmlp=sklearn.metrics.accuracy_score(test_y.reshape(test_X.shape[0]),predilstm)
    bacmlp=sklearn.metrics.balanced_accuracy_score(test_y.reshape(test_X.shape[0]),predilstm)
    a.append(accmlp)
    b.append(bacmlp)

