from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import expr


class AlternatingLeastSquares(object):
    """Train, test and evaluate ALS model"""
    def __init__(self):
        """Instantiate ALS model"""
        self.train = None
        self.test = None

        self.als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating',
                       coldStartStrategy='drop')
        self.model = None
        self.results = []
        self.evaluator = None

    def grid_search(self, train, rank, max_iter, reg_param, num_folds, metric,
                    parallelism):
        """
        Fit a model by running parameter grids

        :param train: spark dataframe, training data
        :param rank: int, rank of ALS
        :param max_iter: int, maxIter of ALS
        :param reg_param: int, regParam of ALS
        :param num_folds: int, numFolds of CrossValidator
        :param metric: str, metric of RegressionEvaluator
        """
        self.train = train.drop('timestamp')
        param_grid = (ParamGridBuilder()
                      .addGrid(self.als.rank, rank)
                      .addGrid(self.als.maxIter, max_iter)
                      .addGrid(self.als.regParam, reg_param)
                      .build())
        self.evaluator = RegressionEvaluator(metricName=metric,
                                             labelCol='rating',
                                             predictionCol='prediction')
        cross_val = CrossValidator(estimator=self.als,
                                   estimatorParamMaps=param_grid,
                                   evaluator=self.evaluator,
                                   numFolds=num_folds,
                                   collectSubModels=True,
                                   parallelism=parallelism)

        self.model = cross_val.fit(train)

        for avg_metric, param_maps in zip(self.model.avgMetrics,
                                          self.model.getEstimatorParamMaps()):
            params = {p.name: round(v, 2) for p, v in param_maps.items()}
            result = {metric: round(avg_metric, 4), **params}
            self.results.append(result)

    def predict(self, test):
        """
        Predict ratings and rankings for the test data

        :param test: spark dataframe, test data
        :return : tuple, dataframes of predicted ratings and rankings
        """

        self.test = test.drop('timestamp')

        # Predict ratings
        self.pred_ratings = self.model.transform(self.test)

        # Predict rankings
        pred = (self.pred_ratings
                .orderBy(['prediction', 'movieId'], ascending=[False, False])
                .groupBy('userId')
                .agg(expr('collect_list(movieId) as predictedRanking'))
                .withColumnRenamed('userId', 'predUserId'))

        truth = (self.pred_ratings
                 .orderBy(['rating', 'movieId'], ascending=[False, False])
                 .groupBy('userId')
                 .agg(expr('collect_list(movieId) as userRanking')))

        self.pred_rankings = (truth.join(pred,
                                         truth['userId'] == pred['predUserId'])
                              .drop('predUserId'))

        return (self.pred_ratings.withColumnRenamed('prediction',
                                                    'predictedRating'),
                self.pred_rankings)

    def rmse(self):
        """
        Calculate RMSE for the predicted ratings

        :return : int, RMSE
        """
        return self.evaluator.evaluate(self.pred_ratings)

    def precision_at_k(self, k):
        """
        Calculate precision at k for the predicted rankings

        :param k: int, calculate precision at k
        :return : int, precision
        """
        rank = self.pred_rankings.rdd.map(lambda tup: (tup[2], tup[1]))
        metrics = RankingMetrics(rank)
        return metrics.precisionAt(k)
