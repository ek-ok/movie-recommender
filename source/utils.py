from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, substring
import pandas as pd
from numpy import arange
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def create_spark_session():
    """Create spark session"""
    spark = (SparkSession.builder
                         .master('local')
                         .appName('Movie Recommendations')
                         .config('spark.driver.host', 'localhost')
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


def rmse_distribution(input_df, group_by='userId'):
    input_df['squared_error'] = input_df.apply(lambda x: (x['rating'] -
                                               x['predictedRating'])**2
                                               if x['predictedRating'] is not
                                               None else 0,
                                               axis=1)
    input_df_agg_sum = input_df.groupby(by=group_by, as_index=False).sum()
    input_df_agg_count = input_df.groupby(by=group_by, as_index=False).count()
    input_df_agg = pd.merge(input_df_agg_sum, input_df_agg_count, on=group_by)[
                   [group_by, 'squared_error_x', 'squared_error_y']]
    input_df_agg['RMSE_agg'] = input_df_agg.apply(lambda x:
                                                  (x['squared_error_x'] /
                                                   x['squared_error_y'])**0.5,
                                                  axis=1)
    return input_df_agg['RMSE_agg']


def top_k_precision_distribution(input_df, k):
    input_df['precision'] = input_df.apply(lambda x: len(list(
                            set(x['userRanking']) &
                            set(x['predictedRanking'])))/float(k), axis=1)
    return input_df['precision']


def calculate_coverage(input_df):
    unique_movie_cnt = len(input_df['movieId'].unique().tolist())
    recommended_movie = [list(_) for _ in input_df['predictedRanking'].tolist()]
    movie_recommended_flatted = set([item for sublist in recommended_movie for item in sublist])
    unique_recommended_movie_cnt = len(movie_recommended_flatted)
    return float(unique_recommended_movie_cnt) / unique_movie_cnt


def plot_lines(x, y, title, x_lab, y_lab, legend_lab=None):
    """
    Output a line chart

    :param x: pandas Series of independant variable
    :param y: pandas Series of dependant variable
    :param gp_cols: pandas DataFrame of optional grouping category
    :return : pyplot line chart
    """

    fig, ax = plt.subplots(figsize=(6, 6))

   # if gp_cols:
   #     df = pd.concat([x, y, gp_cols], axis=1).reset_index()
   #     for _, subset in df.groupby(gp_cols):
   #         label = str(subset[gp_cols].iloc[0].to_dict())
   #         (subset.plot(kind='line', x=x, y=y, label=label, ax=ax)
   #                .set(xlabel=x.name, ylabel=y.name))
   # else:
        
    ax.plot(x, y)
    step_size = (max(x)-min(x))/(len(x)-1)
    ax.set_xticks(arange(min(x), max(x)+step_size, step_size))
    ax.set_title(title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    if step_size < 1:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if legend_lab:
        ax.legend(legend_lab, loc='center left', bbox_to_anchor=(1, 0.5))


def create_quantile_bucket(metric, nTile=10):
    """
    Output the distribution of the valuation metric

    :param title: chart title
    :param metric: pandas Series containing associated valuation metric per ID
    :param nTile: int, number of buckets to split data
    :return : pyplot chart of the valuation metric per nTile of data
    """

    metric_quantile = pd.Series(pd.qcut(metric.rank(method='first'), nTile, labels=False))
    metric_quantile.name = 'Quantile'
    df = pd.concat([metric_quantile, metric], axis=1).reset_index()
    df_grouped = df.groupby('Quantile').mean().reset_index()
    return df_grouped.iloc[:,2]