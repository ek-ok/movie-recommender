from pyspark.sql.functions import col
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql.functions import expr


class BaseLine(object):
    """
    Train, test and evaluate baseline model. Use mean ratings for RSME and
    recommend the most popular k movies for everyone
    """
    def __init__(self):
        self.train = None
        self.test = None
        self.top_movies = None
        self.pred_ratings = None

    def fit(self, train):
        """
        Fit a model by running parameter grids

        :param train: spark dataframe, training data
        """
        self.train = train
        self.avg_ratings = (train
                            .groupBy('movieId')
                            .agg({'rating': 'avg', 'userId': 'count'})
                            .withColumnRenamed('movieId', 'predMovieId')
                            .withColumnRenamed('avg(rating)', 'avgRating')
                            .withColumnRenamed('count(userId)', 'reviewCount'))

    def predict(self, test):
        """
        Predict ratings and rankings for the test data

        :param test: spark dataframe, test data
        :return : tuple, dataframes of predicted ratings and rankings
        """
        self.test = test
        keep_cols = ['movieId', 'userId', 'rating', 'avgRating', 'reviewCount']

        # Predict ratings
        self.pred_ratings = (
            test
            .join(self.avg_ratings,
                  test['movieId'] == self.avg_ratings['predMovieId'],
                  how='left')
            .select(keep_cols)
        )

    def rmse(self):
        """
        Calculate RMSE for the predicted ratings

        :return : int, RMSE
        """
        metrics = RegressionMetrics(
                self.pred_ratings.rdd.map(lambda tup: (tup[2], tup[3])))
        return metrics.rootMeanSquaredError

    def precision_at_k(self, k):
        """
        Calculate precision at k for the predicted rankings

        :param k: int, calculate precision at k
        :return : int, precision
        """
        # Predict rankings
        user_rankings = (
            self.test.orderBy(['rating', 'movieId'], ascending=[False, False])
            .groupBy('userId')
            .agg(expr('collect_list(movieId) as topMovies'))
        )
        top_movies = (self.avg_ratings
                      .filter(col('avgRating') >= 4)
                      .orderBy('reviewCount', ascending=False))

        top_k_movies = (top_movies
                        .limit(k)
                        .agg(expr('collect_list(predMovieId) as topMovies'))
                        .first()
                        .asDict()['topMovies'])

        self.pred_rankings = user_rankings.rdd.map(
                lambda tup: (tup[1], top_k_movies[:len(tup[1])+1]))
        metrics = RankingMetrics(self.pred_rankings)
        return metrics.precisionAt(k)
