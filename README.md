# Recommendation-Engine
Recommnedation Engine Using Keras
The dataset consists of user id, post id and the interaction specific user had with the post i.e Like, Share, Share, Download. 
The model assigns latent factors to each post and user and estimates the corresponding rating. Due to large number of records the 
model cannot predict all records in single numpy array and running loops is very slow. Therefore it splits the the dataset into
multiple arrays and writes on disk. The predict function is then called from command line with file part number in the argument.
The function then reads the corresponding part from the disk, estimate the rating and writes back the dataframe with estimated
rating on the disk. This effectively allows multiprocessing, sppeds up the process with existing compute capabilities.
