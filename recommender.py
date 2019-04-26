#!/usr/bin/env python
# coding: utf-8



from keras.models import load_model
from keras import regularizers
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, Concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from sqlalchemy import create_engine

class read_train_model():
    
    def __init__(self,sql_query,sql_connection):
        self.sql_query=sql_query
        self.sql_connection=sql_connection
        self.df_in=pd.DataFrame()
        self.df_train=pd.DataFrame()
        self.df_test=pd.DataFrame()
        self.n_users=0
        self.n_posts=0
        self.max_user=0
        self.max_post=0
        
    def read_data(self):
        
        engine = create_engine(self.sql_connection, echo=False)
        self.SQL_Query = pd.read_sql_query(self.sql_query, engine)
        
        self.df_in = pd.DataFrame(self.SQL_Query, columns=['UserId','PostId','Likes','Shares','Comments','Downloads','Views'])
        self.df_in['Rating']=self.df_in['Likes']+self.df_in['Comments']+self.df_in['Shares']+self.df_in['Downloads']+self.df_in['Views']
        self.df_in.drop(['Likes','Comments','Shares','Downloads','Views'],axis=1,inplace=True)
 
        self.df_train,self.df_test =train_test_split(self.df_in, test_size = 0.1,random_state = 42 )
        
        self.n_users = len(self.df_in.UserId.unique()) 
        self.n_posts = len(self.df_in.PostId.unique())
        self.max_user,self.max_post=max(self.df_in.UserId),max(self.df_in.PostId)
        

    def define_model(self):
        post_input = Input(shape=[1], name="post-Input")
        post_embedding = Embedding(self.max_post+1,10,  name="post-Embedding")(post_input)
        
        post_vec = Flatten(name="Flatten-post")(post_embedding)

        user_input = Input(shape=[1], name="User-Input")
        user_embedding = Embedding(self.max_user+1, 10, name="User-Embedding")(user_input)
        
       
        user_vec = Flatten(name="Flatten-Users")(user_embedding)

        product_layer = Dot(name="Dot",axes=1)([post_vec, user_vec])

        
        output_connected_layer = Dense(1,activation ='linear')(product_layer)

        model = Model([user_input, post_input],output_connected_layer)
        model.compile(loss='mse', optimizer='adam', metrics=["mae"])
        return model
    
    def train_model(self):
        model =self.define_model()
        training = model.fit([self.df_train.UserId, self.df_train.PostId], self.df_train.Rating, epochs= 30, verbose=1)
        model.save('recommender_model.h5')
        return history
    
    def get_model(self):
        model = load_model('recommender_model.h5')
        print('model loaded')
        return model
    def predict(self):

        model=self.get_model()
        posts=self.df_in['PostId'].unique()
        users=self.df_in['UserId'].unique()
        users_index=np.repeat(users,len(posts))
        posts_index=np.tile(posts,len(users))
        
        split_factor=(len(users_index)//100000)+1
        
        posts_array=np.array_split(posts_index,split_factor)
        users_array=np.array_split(users_index,split_factor)
        
        est=[]
        for i in range(split_factor):
            est_current=model.predict([users_array[i],posts_array[i]])
            est.append(est_current)
            print(str(i)+' of '+str(split_factor-1))
        
        est1=np.concatenate( est, axis=0 )
        est1=est1.reshape(len(est1))
        df_final=pd.DataFrame({'UserId':users_index,'PostId':posts_index,'EstimatedRating':est1})
       
        return df_final
    def predict_2(self):
        model=self.get_model()
        engine = create_engine(self.sql_connection, echo=False)
        post= self.df_in.PostId.unique()
        user= self.df_in.UserId.unique()
        #pids=np.random.choice(post, len(post)//4)
        users_=np.repeat(user,len(post))
        posts_=np.tile(post,len(user))
        
        split_factor=(len(users_)//100000)+1
        
        posts_arr=np.array_split(posts_,split_factor)
        users_arr=np.array_split(users_,split_factor)
        
        esti=[]
        for i in range(split_factor):
            est_curr=model.predict([users_arr[i],posts_arr[i]])
            esti.append(est_curr)
            print(str(i)+' of '+str(split_factor-1))
        
        est11=np.concatenate(esti, axis=0 )
        est11=est11.reshape(len(est11))
        final_df=pd.DataFrame({'UserId':users_,'PostId':posts_,'EstimatedRating':est11})
        df_sorted=final_df.groupby(["UserId"]).apply(lambda x: x.sort_values(["EstimatedRating"], ascending = False)).reset_index(drop=True)
        final_data=df_sorted.groupby(['UserId'])['UserId','PostId','EstimatedRating'].head(100)
        try:
            final_data.to_sql('users_posts_', con=engine ,if_exists='replace')
        except:
            print('error :',sys.exc_info()[0])
            raise
        
    

sql_query= sys.argv[1]
sql_connection= sys.argv[2]

rec_model=read_train_model(sql_query,sql_connection)
rec_model.read_data()
rec_model.define_model()
rec_model.train_model()
sorted_data=rec_model.predict_2()


