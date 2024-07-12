#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')


# In[3]:


car_df.head(10)


# In[4]:


car_df.tail(10)


# In[5]:


sns.pairplot(car_df)


# In[6]:


X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)


# In[7]:


y = car_df['Car Purchase Amount']


# In[8]:


y


# In[9]:


X.shape


# In[10]:


y.shape


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[12]:


X_scaled


# In[13]:


scaler.data_max_


# In[14]:


scaler.data_min_


# In[15]:


y = y.values.reshape(-1,1)


# In[16]:


y_scaled = scaler.fit_transform(y)


# In[17]:


y_scaled


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)


# In[19]:


X_scaled.shape


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[22]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(25, input_dim = 5, activation = 'relu'))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


# In[23]:


model.summary()


# In[24]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[25]:


epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, verbose = 1, validation_split = 0.2)


# In[27]:


epochs_hist.history.keys()


# In[30]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel(' Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[38]:


# Gender, Age, Annual Salary, Credit Card Debt, Net Worth
X_test = np.array([[1, 50, 50000, 10000, 10000]])
y_predict = model.predict(X_test)


# In[39]:


print('Expected Purchase Amount', y_predict)


# In[ ]:




