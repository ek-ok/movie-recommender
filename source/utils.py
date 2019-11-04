from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, substring
import pandas as pd

def create_spark_session():
    """Create spark session"""
    spark = (SparkSession.builder
                         .master('local')
                         .appName('Movie Recommendations')
                         .config('spark.executor.memory', '8g')
                         .config('spark.driver.memory', '8g')
                         .getOrCreate())
    return spark


def prepare_data(spark, sample_size):
    """
    Clean up raw data and split the data into train and split

    :param spark: SparkSession
    :param sample_size: int, number of data size to select
    :return : train, test, both of them are spark dataframes
    """
    movies = spark.read.load('data/ml-20m/movies.csv', format='csv', sep=',',
                             inferSchema='true', header='true')
    ratings = spark.read.load('data/ml-20m/ratings.csv', format='csv', sep=',',
                              inferSchema='true', header='true')

    """
    take subset of database (can't do ordering here as it is really slow over
    a distributed database)
    """
    n_ratings = ratings.limit(sample_size)

    # remove movies and users 1 rating
    user_filter = (n_ratings.groupBy('userId')
                            .agg(count('userId').alias('count'))
                            .filter(col('count') == 1)
                            .select('userId'))
    movie_filter = (n_ratings.groupBy('movieId')
                             .agg(count('movieId').alias('count'))
                             .filter(col('count') == 1)
                             .select('movieId'))
    n_ratings = n_ratings.join(user_filter, ['userId'], how='left_anti')
    n_ratings = n_ratings.join(movie_filter, ['movieId'], how='left_anti')

    # movies with valid genre
    movies_genre = movies.filter(col('genres') != '(no genres listed)')
    # remove year
    movies_genre = movies_genre.withColumn('year',
                                           substring(col('title'), -5, 4))
    genre_filter = movies_genre.select('movieId')

    # keep only movies with genre
    n_ratings = n_ratings.join(genre_filter, ['movieId'], how='left_semi')

    # test train split
    train, test = n_ratings.randomSplit([0.8, 0.2], seed=12345)

    # take union set of users, movies in both data pieces
    train = train.join(test.select('userId'), ['userId'], how='left_semi')
    train = train.join(test.select('movieId'), ['movieId'], how='left_semi')

    test = test.join(train.select('userId'), ['userId'], how='left_semi')
    test = test.join(train.select('movieId'), ['movieId'], how='left_semi')

    return train, test
    
def RMSE_distribution(input_df, group_by = 'userId'):
    input_df['squared_error'] = input_df.apply(lambda x: (x['rating'] - x['predictedRating'])**2 if x['predictedRating'] is not None else 0, axis=1)
    input_df_agg_sum = input_df.groupby(by=group_by, as_index=False).sum()
    input_df_agg_count = input_df.groupby(by=group_by, as_index=False).count()
    input_df_agg = pd.merge(input_df_agg_sum,input_df_agg_count,on=group_by)[[group_by,'squared_error_x','squared_error_y']]
    input_df_agg['RMSE_agg'] = input_df_agg.apply(lambda x: (x['squared_error_x']/x['squared_error_y'])**0.5, axis=1)
    return input_df_agg['RMSE_agg']
    
def top_k_precision_distribution(input_df, k):
    input_df['precision'] = input_df.apply(lambda x: len(list(set(x['userRanking']) & set(x['predictedRanking'])))/float(k), axis=1)
    return input_df['precision']
