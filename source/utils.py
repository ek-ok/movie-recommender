from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, substring
import pandas as pd
import matplotlib.pyplot as plt

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


def plotLines(title, x, y, gp_cols=None):
    """
    Output a line chart

    :param x: pandas Series of independant variable
    :param y: pandas Series of dependant variable
    :param gp_cols: pandas DataFrame of optional grouping category
    :return : pyplot line chart
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if gp_cols:
        df = pd.concat([x, y, gp_cols], axis=1).reset_index()
        for _, subset in df.groupby(gp_cols):
            label = str(subset[gp_cols].iloc[0].to_dict())
            (subset.plot(kind='line', x=x, y=y, label=label, ax=ax)
                   .set(xlabel=x.name, ylabel=y.name)) 
    else:
        ax.plot(x, y)
        
    ax.title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.show()

    
def plotDistribution(title, metric, nTile=10):
    """
    Output the distribution of the valuation metric

    :param title: chart title
    :param metric: pandas Series containing associated valuation metric per ID
    :param nTile: int, number of buckets to split data
    :return : pyplot chart of the valuation metric per nTile of data
    """

    metricTileLabel = pd.Series(pd.qcut(metric, nTile, labels=False))
    metricTileLabel.name = 'NTile'
    df = pd.concat([metricTileLabel, metric], axis=1).reset_index()
    df_grouped = df.groupby('NTile').mean().reset_index()
    plotLines(title, metricTileLabel, df_grouped.columns([1]))
