#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 05:57:30 2019

@author: gaurav
"""

from keras.models import load_model
from keras import regularizers
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, multiply, Concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


class read_train_model():
    
    def __init__(self,data_path):
        self.data_path=data_path
        self.df_in=pd.DataFrame()
        self.df_train=pd.DataFrame()
        self.df_test=pd.DataFrame()
        self.n_user=0
        self.n_posts=0
        self.model
        
    def read_data(self,sql_query,db_connection):
        self.df_in = pd.read_sql_query(sql_query,db_connection)
        
        self.df_in['Rating']=self.df_in['Likes']+self.df_in['Comments']+self.df_in['Shares']+self.df_in['Downloads']+self.df_in['Views']
        self.df_in.drop(['Likes','Comments','Shares','Downloads','Views'],axis=1,inplace=True)
    
        self.df_train,self.df_test =train_test_split(self.df_in, test_size = 0.1,random_state = 42 )

        self.n_users = len(self.df_in.UserId.unique()) 
        self.n_posts = len(self.df_in.PostId.unique())
        self.max_user,self.max_post=max(self.df_in.UserId),max(self.df_in.PostId)
        

    def define_model(self):
        post_input = Input(shape=[1], name="post-Input")
        post_embedding = Embedding(self.max_post+1,100,  name="post-Embedding")(post_input)
        lp = Dense(100,activation = 'relu',kernel_regularizer=regularizers.l2(0.001),)(post_embedding)
        Dropout(0.4)
        post_vec = Flatten(name="Flatten-post")(lp)

        user_input = Input(shape=[1], name="User-Input")
        user_embedding = Embedding(self.max_user+1, 100, name="User-Embedding")(user_input)
        l2 = Dense(100,activation = 'relu',kernel_regularizer=regularizers.l2(0.001))(user_embedding)
        Dropout(0.4)
        user_vec = Flatten(name="Flatten-Users")(l2)

        product_layer = Concatenate(name="Concat",)([post_vec, user_vec])

        fully_connected_layer = Dense(200,activation ='relu')(product_layer)
        fully_connected_layer_2 = Dense(100,activation ='relu')(fully_connected_layer)
        fully_connected_layer_3 = Dense(100,activation ='relu')(fully_connected_layer_2)
        fully_connected_layer_4 = Dense(100,activation ='relu')(fully_connected_layer_3)


        output_connected_layer = Dense(1,activation ='linear')(fully_connected_layer_4)

        model = Model([user_input, post_input],output_connected_layer)
        model.compile(loss='mse', optimizer='adam', metrics=["mse"])
        return model
    
    def train_model(self):
        model =self.define_model()
        history = model.fit([self.df_train.UserId, self.df_train.PostId], self.df_train.Rating,validation_split=0.1 , epochs= 1, verbose=1)
        self.model=history
        model.save('recommender_model.h5')
        return history
    
    def get_model(self):
        model = load_model('recommender_model.h5')
        print('model loaded')
        return model

sql_query=sys.argv[1]
db_connection=sys.argv[2]
rec_model=read_train_model(sql_query,db_connection)
rec_model.read_data()
rec_model.define_model()
rec_model.train_model()


    def predict
posts=rec_model.df_in['PostId'].unique()
users=rec_model.df_in['UserId'].unique()
users_index=np.repeat(users,len(posts))
posts_index=np.tile(posts,len(users))


a=np.split(posts_index,10)
b=np.split(users_index,10)
for i in range(10):
    name='data_part'+str(i)
    filename=name+'.csv'
    name=pd.DataFrame({'Users':b[i],'Posts':a[i]})
    
    name.to_csv(filename)

final_df.to_sql(target_table_name,db_connection,if_exists='replace')