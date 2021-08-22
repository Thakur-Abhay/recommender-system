import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import pdb
from scipy.spatial.distance import cosine,correlation


# print (pd._version)
#reading MovieLens dataset
# path='C:\Users\\abhay\\Dropbox\\My PC (LAPTOP-CPO2A5NR)\\Desktop\AT\\Recommender syst
# em\\ml-latest-small'
# path=r'C:\Users\abhay\Dropbox\My PC (LAPTOP-CPO2A5NR)\Desktop\AT\Recommender system\ICICIBANK1.xlsx'
# # path=r'C:\Users\abhay\Dropbox\My PC (LAPTOP-CPO2A5NR)\Desktop\AT\Recommender system\df1.txt'
# df=pd.read_excel(path)

# print(df.head())
rating_df=pd.read_csv(r'C:\Users\abhay\Dropbox\My PC (LAPTOP-CPO2A5NR)\Desktop\AT\Recommender system\ml-latest-small\ratings.csv')
# pdb.set_trace()
# print('one')
# print('two')
# #print(rating_df.head())
rating_df.drop('timestamp',axis=1,inplace=True)
# print(rating_df.head())
lenuser=len(rating_df.userId.unique())
# print(lenuser)
lenmovie=len(rating_df.movieId.unique())
# print(lenmovie)
#now we create a df containing only the userid and the movieid with the rating
user_movie_df=rating_df.pivot(index='userId',columns='movieId',values='rating')
# print(user_movie_df)

user_movie_df.fillna(0,inplace=True)
# print(user_movie_df)
#the below command is making the choice in creating the similar users df ..the pairwise distances function basically
#does all the work for me and is an example of a supervised learning function
#it creates a new matrix which is n*n matrix of every user with every user and the closer two users are to 1
#the more similar their choices are going to be and hence we can predict the choice of a user by eeing the choice
#of the use rsimmilar to him in the matrix

user_similar=1-pairwise_distances(user_movie_df.values,metric='cosine')
user_similar_df=pd.DataFrame(user_similar)
# print(user_similar_df)
user_similar_df.index=rating_df.userId.unique()
user_similar_df.columns=rating_df.userId.unique()
# print(user_similar_df)
np.fill_diagonal(user_similar,0)
# print(user_similar_df)
# print(user_similar_df.idxmax(axis=1)[0:5])
movies_df=pd.read_csv(r'C:\Users\abhay\Dropbox\My PC (LAPTOP-CPO2A5NR)\Desktop\AT\Recommender system\ml-latest-small\movies.csv')
# print(movies_df.head())
# print(len(movies_df))
movies_df.drop('genres',axis=1,inplace=True)
# print(movies_df.head())

def get_user_similar_movies(user1,user2):
    common_movies=rating_df[rating_df.userId==user1].merge(rating_df[rating_df.userId==user2],on='movieId',how='inner')
    return common_movies.merge(movies_df,on='movieId')
common_movies=get_user_similar_movies(2,338)
# print(common_movies[(common_movies.rating_x>=3.0) & (common_movies.rating_y>=3.0)])
common_movies=get_user_similar_movies(2,333)
# print((common_movies))
# rating_matrix=rating_df.pivot(index='movieId',columns='userId',values='rating').reset_index(drop=True)
# rating_matrix.fillna(0,inplace=True)
# #this line finds the relation btw movies ont he basis of item based collaborative filtering
# movie_sim=1-pairwise_distances(rating_matrix.values,metric='correlation')
# movie_sim_df=pd.DataFrame(movie_sim)
# print(movie_sim_df)
# print(movie_sim_df.shape)
rating_mat=rating_df.pivot(index='movieId',columns='userId',values='rating').reset_index(drop=True)
rating_mat.fillna(0,inplace=True)
# print((rating_mat))
#pdb.set_trace()
#Nrating_mat1=rating_mat.resize(1,5)

movie_sim=pairwise_distances(user_movie_df.values,metric="correlation")

movie_sim=1-pairwise_distances(rating_mat.values,metric="correlation")
# print('crossed movie_sim successfuly')
# pdb.set_trace()
movie_sim_df=pd.DataFrame(movie_sim)
print(movie_sim_df.head())
def get_similar_movies(movieid,topN=5):
	#getting the index of the movie record in movies_data frame
	movieidx=movies_df[movies_df.movieId==movieid].index[0]
	movies_df['similarity']=movie_sim_df.iloc[movieidx]
	top_n=movies_df.sort_values(['similarity'],ascending=False)[0:topN]
	return top_n

# print(get_similar_movies(858))
ans=pd.DataFrame(get_similar_movies(858))
print(ans)



