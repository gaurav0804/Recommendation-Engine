#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:41:55 2019

@author: msf
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


class read_train_model():
    
    def __init__(self,data_path):
        self.data_path=data_path
        self.df_in=pd.DataFrame()
        self.df_train=pd.DataFrame()
        self.df_test=pd.DataFrame()
        self.n_user=0
        self.n_posts=0
        
    def read_data(self):
        self.df_in = pd.read_csv(self.data_path)
        self.df_in.loc[self.df_in.Likes>0,'Likes']=1
        self.df_in.loc[self.df_in.Comments>0,'Comments']=1
        self.df_in.loc[self.df_in.Shares>0,'Shares']=1
        self.df_in.loc[self.df_in.Downloads>0,'Downloads']=1
        self.df_in.loc[self.df_in.Views>0,'Views']=1

        self.df_in['Rating']=self.df_in['Likes']+self.df_in['Comments']+self.df_in['Shares']+self.df_in['Downloads']+self.df_in['Views']
        self.df_in.drop(['Likes','Comments','Shares','Downloads','Views'],axis=1,inplace=True)
    
        self.df_in.UserId = self.df_in.UserId.astype('category').cat.codes.values
        self.df_in.PostId = self.df_in.PostId.astype('category').cat.codes.values

        self.df_train,self.df_test =train_test_split(self.df_in, test_size = 0.1,random_state = 42 )

        self.n_users = len(self.df_in.UserId.unique()) 
        self.n_posts = len(self.df_in.PostId.unique())
        
    
    
    def user_post_ids(self):
        dff = pd.read_csv(self.data_path)
        dff.loc[dff.Likes>0,'Likes']=1
        dff.loc[dff.Comments>0,'Comments']=1
        dff.loc[dff.Shares>0,'Shares']=1
        dff.loc[dff.Downloads>0,'Downloads']=1
        dff.loc[dff.Views>0,'Views']=1

        dff['Rating']=dff['Likes']+dff['Comments']+dff['Shares']+dff['Downloads']+dff['Views']
        dff.drop(['Likes','Comments','Shares','Downloads','Views'],axis=1,inplace=True)
        trainn,testt = train_test_split(dff,test_size = 0.1,random_state =42)
        
        return trainn, testt


    def define_model(self):
        post_input = Input(shape=[1], name="post-Input")
        post_embedding = Embedding(self.n_posts+1,10,  name="post-Embedding")(post_input)
        lp = Dense(10,activation = 'relu',kernel_regularizer=regularizers.l2(0.001),)(post_embedding)
        Dropout(0.4)
        post_vec = Flatten(name="Flatten-post")(lp)

        user_input = Input(shape=[1], name="User-Input")
        user_embedding = Embedding(self.n_users+1, 10, name="User-Embedding")(user_input)
        l2 = Dense(10,activation = 'relu',kernel_regularizer=regularizers.l2(0.001))(user_embedding)
        Dropout(0.4)
        user_vec = Flatten(name="Flatten-Users")(l2)

        product_layer = Dot(name="Dot-Product", axes=1)([post_vec, user_vec])

        fully_connected_layer = Dense(10,activation ='relu')(product_layer)
        fully_connected_layer_2 = Dense(10,activation ='relu')(fully_connected_layer)
        fully_connected_layer_3 = Dense(10,activation ='relu')(fully_connected_layer_2)
        fully_connected_layer_4 = Dense(10,activation ='relu')(fully_connected_layer_3)


        output_connected_layer = Dense(1,activation ='linear')(fully_connected_layer_4)

        model = Model([user_input, post_input],output_connected_layer)
        model.compile(loss='mse', optimizer='adam', metrics=["mae"])
        return model
    
    def train_model(self):
        model =self.define_model()
        history = model.fit([self.df_train.UserId, self.df_train.PostId], self.df_train.Rating,validation_split=0.1 , epochs= 1, verbose=1)
        model.save('recommender_model.h5')
        return history
    
    def get_model(self):
        model = load_model('recommender_model.h5')
        print('model loaded')
        return model
    
    
    def get_estimation_data(self):
        def duplicate(testList,n ): 
            return list(testList*n)
                
        n_users,n_posts,train,test=self.n_user,self.n_posts,self.df_train,self.df_test
        trainn,testt=self.user_post_ids()
        len_post = len(test.PostId.unique())
        len_user= len(testt.UserId.unique())
        p = test.PostId.unique()
        unique_postids = p.tolist()
        upids=duplicate(unique_postids,len_user) #post_ids_looped


        u =test.UserId.unique()
        unique_userids =u.tolist()
        un = np.array(unique_userids)
        user_loop =np.repeat(unique_userids,len_post) #user_ids_looped
        ttpids = testt['PostId'].unique()
        ttuid = testt['UserId'].unique()
        pp = testt.PostId.unique()
        uunique_postids = pp.tolist()
        uupids=duplicate(uunique_postids,len_user) #post_ids_looped


        uu =testt.UserId.unique()
        uunique_userids =uu.tolist()
        uun = np.array(uunique_userids)
        uuser_loop =np.repeat(uunique_userids,len_post) #user_ids_looped
        post_data = np.array(upids)
        user = np.array(user_loop)
        model=self.get_model()
        estimations = model.predict([user, post_data]) #predictions
   
        pid =pd.DataFrame(uupids)  #forming dataframes
        uid =pd.DataFrame(uuser_loop)
        estimation =pd.DataFrame(estimations)
        dataa = pd.merge(estimation,pid,left_index =True,right_index = True)
        data = pd.merge(dataa,uid,left_index = True, right_index= True)
        data.rename(columns={'0_x':'estimation','0_y':'post_id',0:'user_id'},inplace = True)
        final_data_sorted = data.groupby(["user_id"]).apply(lambda x: x.sort_values(["estimation"], ascending = False)).reset_index(drop=True)
        return final_data_sorted
   


rec_model=read_train_model('/home/gaurav/Desktop/RecEng/ML-DataSet/Wall_Activity_User_Post.csv')
rec_model.read_data()
rec_model.define_model()
rec_model.train_model()



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
    
