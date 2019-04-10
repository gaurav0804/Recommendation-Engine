#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 07:04:15 2019

@author: gaurav
"""
#1 process everything at once takes 205 s on small data set
import time
posts=rec_model.df_in['PostId'].unique()
users=rec_model.df_in['UserId'].unique()

model=rec_model.get_model()


start = time.time()

users_index=np.repeat(users,len(posts))
posts_index=np.tile(posts,len(users))
estimations=model.predict([users_index,posts_index])
estimations=np.reshape(estimations,(len(estimations),))
final_df=pd.DataFrame({'Users':users_index,'Posts':posts_index,'EastimatedRating':estimations})
print("Complete")
end = time.time()
print('total time (s)= ' + str(end-start))


#2 multiprocessing 239 seconds
import time
from joblib import Parallel, delayed
import multiprocessing

posts=rec_model.df_in['PostId'].unique()
users=rec_model.df_in['UserId'].unique()
final_df=pd.DataFrame(index=users)
model=rec_model.get_model()

i=0

def estimate(pid):
    post=np.full(shape=len(users),fill_value=pid)
    est=model.predict([users,post])
    final_df[pid]=est
    return pid


num_cores = multiprocessing.cpu_count()
start=time.time()
result=Parallel(n_jobs=-1,verbose=5,timeout=1000,backend="multiprocessing")(delayed(estimate)(p) for p in posts)
print(result)
end=time.time()
print('time taken:'+str(end-start))

#3 loop over one posts at a time takes 300 sec
import time
posts=rec_model.df_in['PostId'].unique()
users=rec_model.df_in['UserId'].unique()
final_df=pd.DataFrame(index=users)
model=rec_model.get_model()


i=1
start = time.time()
for pid in posts:
    rec_posts_uid=[]
    post=np.full(shape=len(users),fill_value=pid)
    est=model.predict([users,post])
    final_df[pid]=est
    print(str(i)+':'+str(pid),end=',')
    i+=1
print("Complete")
end = time.time()
print('total time (s)= ' + str(end-start))

#4 loop over 4 posts at a time 295 seconds
import time
posts=rec_model.df_in['PostId'].unique()
users=rec_model.df_in['UserId'].unique()

model=rec_model.get_model()
splits=len(posts)//4+1
posts_split_array=np.array_split(posts,splits)
estimations=np.empty([0,1])

i=1
start = time.time()
for splits in posts_split_array:
    repeat_count=len(splits)
    posts_array=np.repeat(splits,len(users))
    users_array=np.tile(users,repeat_count)
    est=model.predict([users_array,posts_array])
    estimations=np.append(estimations,est)
    print(str(i)+':'+str(splits),end=',')
    i+=1
users_index=np.repeat(users,len(posts))
posts_index=np.tile(posts,len(users))

final_df=pd.DataFrame({'Users':users_index,'Posts':posts_index,'Rating':estimations})
print("Complete")
end = time.time()
print('total time (s)= ' + str(end-start))
