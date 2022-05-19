#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/water', methods=['POST', 'GET'])
def water():
    return render_template('result.html')

@app.route('/result.html',methods=['POST','GET'])
def water_potable():
    int_features =[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    if prediction == 0:
        pred = 'This water sample is Not drinkable'
    elif prediction ==1:
        pred = 'This water sample is safe to drink'
    output=pred
    
    return render_template('result.html', prediction_text='Water potability : {}'.format(output))

    
if __name__ =="__main__":
    app.run(debug=True)

