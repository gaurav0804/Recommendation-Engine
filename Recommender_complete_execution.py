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
import sqlalchemy
from sqlalchemy import create_engine
import MySQLdb
import time
import gc



class read_train_model():
    
    def __init__(self,sql_query):
        self.sql_query=sql_query
        self.df_in=pd.DataFrame()
        self.df_train=pd.DataFrame()
        self.df_test=pd.DataFrame()
        self.n_users=0
        self.n_posts=0
        self.max_user=0
        self.max_post=0
        self.df_final=pd.DataFrame()
        
    def read_data(self):
        
        engine = create_engine('mysql+mysqldb://root:stick@52r@127.0.0.1:3306/ML', echo=False)
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
        post_embedding = Embedding(self.max_post+1,20,  name="post-Embedding")(post_input)
        lp = Dense(20,activation = 'relu',kernel_regularizer=regularizers.l2(0.001),)(post_embedding)
        Dropout(0.4)
        post_vec = Flatten(name="Flatten-post")(lp)

        user_input = Input(shape=[1], name="User-Input")
        user_embedding = Embedding(self.max_user+1, 20, name="User-Embedding")(user_input)
        l2 = Dense(20,activation = 'relu',kernel_regularizer=regularizers.l2(0.001))(user_embedding)
        Dropout(0.4)
        user_vec = Flatten(name="Flatten-Users")(l2)

        product_layer = Dot(name="Dot",axes=1)([post_vec, user_vec])

        fully_connected_layer = Dense(40,activation ='relu')(product_layer)
        fully_connected_layer_2 = Dense(40,activation ='relu')(fully_connected_layer)
        fully_connected_layer_3 = Dense(40,activation ='relu')(fully_connected_layer_2)
        fully_connected_layer_4 = Dense(40,activation ='relu')(fully_connected_layer_3)


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
    def predict(self):
        model=self.get_model()
        posts=self.df_in['PostId'].unique()
        users=self.df_in['UserId'].unique()
        users_index=np.repeat(users,len(posts))
        posts_index=np.tile(posts,len(users))
        
        split_factor=(len(users_index)//100000)+1
        engine = create_engine('mysql+mysqldb://root:stick@52r@127.0.0.1:3306/ML', echo=False)
        posts_array=np.array_split(posts_index,split_factor)
        users_array=np.array_split(users_index,split_factor)
        
        est=[]
        st=time.time()
        for i in range(split_factor):
            est_current=model.predict([users_array[i],posts_array[i]])
            est.append(est_current)
            est1=np.concatenate(est_current, axis=0 )
            est1=est1.reshape(len(est1))
            df=pd.DataFrame({'UserId':users_array[i],'PostId':posts_array[i],'EstimatedRating':est1})
            if i==0:
                df.to_sql('users_posts_pred', con=engine ,if_exists='replace',chunksize = 10000)
                del df
                gc.collect()
                print(str(i)+' of '+str(split_factor))
            else:
                df.to_sql('users_posts_pred', con=engine ,if_exists='append',chunksize = 10000)
                del df
                gc.collect()
                print(str(i)+' of '+str(split_factor))
        #est1=np.concatenate( est, axis=0 )
        #est1=est1.reshape(len(est1))
        #self.df_final=pd.DataFrame({'UserId':users_index,'PostId':posts_index,'EstimatedRating':est1})
        en=time.time()
        print('Predicted and written in df in: '+str(en-st))
        #df_final.to_csv('/home/msf/final_output.csv')
    #def write_to_sql(self):
        #engine = create_engine('mysql+mysqldb://root:stick@52r@127.0.0.1:3306/ML', echo=False)
        #st=time.time()
        #df_final=self.df_final
        #try:
            #df_final.to_sql('users_posts_pred', con=engine ,if_exists='replace',chunksize = 10000)
           #del df_final
            #gc.collect()
        #except:
            #print('error :',sys.exc_info()[0])
            #raise
        #finally:
            #en=time.time()
           #print('total time to write SQL :'+str(en-st))
        

sql_query= 'select * from WallData'#sys.argv[1]
rec_model=read_train_model(sql_query)
rec_model.read_data()
rec_model.define_model()
rec_model.train_model()
rec_model.predict()
#rec_model.write_to_sql()
