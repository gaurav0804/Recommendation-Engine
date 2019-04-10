#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:41:55 2019

@author: msf
"""
import numpy as np
import pandas as pd
from keras.models import load_model
import sys
part_num=sys.argv[1]
data_path='/home/gaurav/Desktop/RecEng/data_part'+str(part_num)+'.csv'
df=pd.read_csv(data_path)

model=load_model('/home/gaurav/Desktop/RecEng/recommender_model.h5')
print('model loaded')
user_array=np.array(df['Users'])
post_array=np.array(df['Posts'])

estimation=model.predict([user_array,post_array])

df['Estimates']=estimation
filename='predictions_part'+str(part_num)+'.csv'
df.to_csv(filename)
print('complete')
