import pandas as pd
import numpy as np
from pyspark.sql.window import Window
import findspark
findspark.init()
import pyspark
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import RowMatrix, CoordinateMatrix
import timeit
from .utils import create_spark_session, prepare_data, RMSE_distribution

SIMILARITY_FILE_SORTED = 'similarity_matrix_sorted.parquet'

class neighbor_based_recommender():
    
    # def __init__(self, train_data_file, test_data_file):
        # self.train_data_file = train_data_file
        # self.test_data_file = test_data_file
    def __init__(self, sample_size):
        spark = create_spark_session()
        self.train_data, self.test_data = prepare_data(spark, sample_size)
        self.sample_size = sample_size
    
    def train(self, neighbor_size = 5, recalculate = False):
        start = timeit.default_timer()
        if recalculate:
            self.calculate_similarity()
        self.predict_ratings(neighbor_size)
        stop = timeit.default_timer()
        self.runtime = stop - start
        
    def calculate_similarity(self):
        # spark = (SparkSession
        #      .builder
        #      .master('local')
        #      .appName('Movie Recommendations')
        #      .config('spark.executor.memory', '8g')
        #      .config('spark.driver.memory', '8g')
        #      .getOrCreate())
        # train = spark.read.parquet(self.train_data_file)
        train = self.train_data
        train_user_mean = train.groupBy("userId").agg(F.mean('rating'))
        train_user_mean = train_user_mean.withColumnRenamed("avg(rating)", "user_mean")
        train_rating_avg = train.join(train_user_mean, 'userId', how='left_outer')
        train_rating_avg = train_rating_avg.select('*', (train_rating_avg.rating - train_rating_avg.user_mean).alias('rating_norm'))
        rdd = train_rating_avg.select('movieId', 'userId', 'rating_norm').rdd.map(tuple)
        coord = CoordinateMatrix(rdd)
        mat = coord.toRowMatrix()
        similarities = mat.columnSimilarities()
        similarities_df = similarities.entries.toDF()
        window = Window.partitionBy(similarities_df['i']).orderBy(similarities_df['value'].desc())
        similarities_df_ranked= similarities_df.select('*', F.row_number().over(window).alias('row_number')) \
          .filter(F.col('row_number') <= 100)
        similarities_df_ranked.write.parquet(SIMILARITY_FILE_SORTED, mode='overwrite')

    def predict_ratings(self, neighbor_size):
        similarity_matrix_sorted = pd.read_parquet(SIMILARITY_FILE_SORTED)
        i_max, j_max = int(similarity_matrix_sorted['i'].max().item()), int(similarity_matrix_sorted['j'].max().item())
        user_max = np.max([i_max, j_max])
        activation_matrix = np.zeros([user_max, user_max])
        for user in similarity_matrix_sorted['i'].unique().tolist():
            user_neighbors = similarity_matrix_sorted[similarity_matrix_sorted['i']==user]
            for neighbor in range(np.min([neighbor_size,user_neighbors.shape[0]])):
                activation_matrix[user-1][user_neighbors[user_neighbors['row_number']==neighbor+1]['j'].values[0]-1] \
                                          = user_neighbors[user_neighbors['row_number']==neighbor+1]['value'].values[0]
        
        # train_data = pd.read_parquet(self.train_data_file)
        train_data = self.train_data.toPandas()
        self.movie_dict = {val:idx for idx,val in enumerate(sorted(train_data['movieId'].unique().tolist()))}
        distinct_user, max_user = train_data['userId'].unique().tolist(), train_data['userId'].max()
        if len(distinct_user) != max_user:
            for seq in range(max_user):
                if seq+1 not in distinct_user:
                    train_data = train_data.append({'userId':seq+1, 'movieId':1, 'rating':0.5, 'timestamp':0} , ignore_index=True)
        Mean = train_data.groupby(by="userId",as_index=False)['rating'].mean()
        train_data_avg = pd.merge(train_data,Mean,on='userId')
        train_data_avg['adg_rating']=train_data_avg['rating_x']-train_data_avg['rating_y']
        train_data_pivot = pd.pivot_table(train_data_avg,values='adg_rating',index='userId',columns='movieId')
        train_data_pivot_user = train_data_pivot.apply(lambda row: row.fillna(row.mean()), axis=1)
        self.result = activation_matrix.dot(train_data_pivot_user.values) + np.tile(Mean['rating'].values,(train_data_pivot_user.shape[1],1)).transpose()

    def recommend_movie(self, user_id, result_cnt = 5):
        return self.result[user_id-1].argsort()[-result_cnt:][::-1]
    
    def rating_prediction(self, user_id, movie_id):
        if int(movie_id) in self.movie_dict:
            return self.result[int(user_id) - 1][self.movie_dict[int(movie_id)]]
        return None
        
    def RMSE(self):
        
        # test_data = pd.read_parquet(self.test_data_file)
        test_data = self.test_data.toPandas()
        test_data['predictedRating'] = test_data.apply(lambda x: self.rating_prediction(x['userId'],x['movieId']), axis=1)
        test_data['squared_error'] = test_data.apply(lambda x: (x['rating'] - x['predictedRating'])**2 if x['predictedRating'] is not None else 0, axis=1)
        RMSE_total = (float(test_data['squared_error'].sum())/test_data[~pd.isna(test_data['predictedRating'])].shape[0])**0.5
        # test_data_user_sum = test_data.groupby(by='userId', as_index=False).sum()
        # test_data_user_count = test_data.groupby(by='userId', as_index=False).count()
        # test_data_user = pd.merge(test_data_user_sum,test_data_user_count,on='userId')[['userId','squared_error_x','squared_error_y']]
        # test_data_user['RMSE_user'] = test_data_user.apply(lambda x: (x['squared_error_x']/x['squared_error_y'])**0.5, axis=1)
        # test_data_user = test_data_user[['userId','RMSE_user']]
        test_data_user = RMSE_distribution(test_data, 'userId')
        # test_data_movie_sum = test_data.groupby(by='movieId', as_index=False).sum()
        # test_data_movie_count = test_data.groupby(by='movieId', as_index=False).count()
        # test_data_movie = pd.merge(test_data_movie_sum,test_data_movie_count,on='movieId')[['movieId','squared_error_x','squared_error_y']]
        # test_data_movie['RMSE_user'] = test_data_movie.apply(lambda x: (x['squared_error_x']/x['squared_error_y'])**0.5, axis=1)
        # test_data_movie = test_data_movie[['movieId','RMSE_user']]
        test_data_movie = RMSE_distribution(test_data, 'movieId')
        return RMSE_total, test_data_user, test_data_movie
    
    def top_k_precision(self, k = 5):
        # test_data = pd.read_parquet(self.test_data_file)
        test_data = self.test_data.toPandas()
        sorted_test_data = test_data.sort_values(['userId','rating'], ascending=[True,False])
        precision = []
        for user in test_data['userId'].unique().tolist():
            movie_to_check_cnt = np.min([k, test_data[test_data['userId']==user].shape[0]])
            movie_to_check = test_data[test_data['userId']==user]['movieId'].tolist()
            user_top_pick = movie_to_check[:movie_to_check_cnt]
            movie_to_check_prediction = [(_,self.rating_prediction(user,_)) for _ in movie_to_check]
            movie_to_check_prediction_sorted = sorted(movie_to_check_prediction, key = lambda x:x[1], reverse = True)
            user_top_pick_prediction, _ = zip(*movie_to_check_prediction_sorted[:movie_to_check_cnt])
            precision.append([user, len(list(set(user_top_pick) & set(user_top_pick_prediction)))/float(k)])
            _, precision_number = zip(*precision)
        return sum(precision_number)/len(precision_number),precision
