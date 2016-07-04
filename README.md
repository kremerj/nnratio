# nnratio
A nearest neighbor density ratio estimator with model selection, written in Python 2. It uses numpy and scikit-learn.

The algorithm is explained in: Jan Kremer, Fabian Gieseke, Kim Steenstrup Pedersen, and Christian Igel. Nearest Neighbor Density Ratio Estimation for Large-Scale Applications in Astronomy. *Astronomy and Computing* **12**:67-72, 2015

```
@Article{kremer15,
  author  = {J. Kremer and F. Gieseke and K. {Steenstrup Pedersen} and C. Igel},
  title   = {Nearest neighbor density ratio estimation for large-scale applications in astronomy},
  journal = {Astronomy and Computing},
  year    = 2015,
  pages   = {67-72},
  volume  = 12,
}
```


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
