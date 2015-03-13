# NN-density-ratio
A nearest neighbor density ratio estimator with model selection, written in Python 2. It uses numpy and scikit-learn.

# Howto
from model.base import NearestNeighborsRatioEstimator
estimator = NearestNeighborsRatioEstimator()
K_list = [2,4,8,16,32]
estimator.fit_cv(training_points,test_points,K_list)
weights = estimator.compute_weights(evaluation_points)
