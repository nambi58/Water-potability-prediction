#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
df = pd.read_csv("water_potability.csv")
df['ph'] = df['ph'].fillna(df.groupby(['Potability'])['ph'].transform('mean'))
df['Sulfate'] = df['Sulfate'].fillna(df.groupby(['Potability'])['Sulfate'].transform('mean'))
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df.groupby(['Potability'])['Trihalomethanes'].transform('mean'))
X = df.drop("Potability", axis=1)
y = df["Potability"]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
pickle.dump(rfc, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

