# nnratio
A nearest neighbor density ratio estimator with model selection, written in Python 2. It uses numpy and scikit-learn.

![Density ratio estimation example](/images/example.png)

## Install
```
git clone https://github.com/kremerj/nnratio.git
cd nnratio
python setup.py install
```

## Use
```
from nnratio import NearestNeighborsRatioEstimator
estimator = NearestNeighborsRatioEstimator()
K_list = [2,4,8,16,32]
estimator.fit_cv(training_points,test_points,K_list)
weights = estimator.compute_weights(evaluation_points)
```
