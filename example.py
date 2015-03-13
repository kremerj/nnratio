import numpy as np
from matplotlib import pyplot as plt
from nnratio import NearestNeighborsRatioEstimator
import seaborn as sns

def generate_data():
    """ generate toy data from two Gaussian distributions
        returns: training sample, test sample, evaluation sample
    """
    n_data = 5000
    n_ev   = 500
    
    cov_tr  = np.array([[1,0],[0,1]])
    mean_tr = np.array(([1.0,1.0]))
    
    cov_te  = np.array([[1,0],[0,1]])
    mean_te = np.array([0.5,0.5])
    
    np.random.seed(0)
    
    L_tr = np.linalg.cholesky(cov_tr)
    L_te = np.linalg.cholesky(cov_te)
    
    x_tr = np.dot(L_tr,(np.random.randn(n_data,2) + mean_tr).T).T
    x_te = np.dot(L_te,(np.random.randn(n_data,2) + mean_te).T).T
    
    x_ind = np.linspace(-2,4,num=int(np.sqrt(n_ev)))
    y_ind = np.linspace(-2,4,num=int(np.sqrt(n_ev)))
    
    x_ev = np.transpose([np.tile(x_ind, len(y_ind)), np.repeat(y_ind, len(x_ind))])
    
    return x_tr, x_te, x_ev

x_tr, x_te, x_ev = generate_data()

# perform nearest neigbhor ratio estimation with automatic model selection
estimator = NearestNeighborsRatioEstimator()
K_list = [2,4,8,16,32]
estimator.fit_cv(x_tr,x_te,K_list)
weights = estimator.compute_weights(x_ev)

# plot ratio estimates
fig,ax = plt.subplots(1,2, figsize=(14,6))
cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True)
plt.suptitle('Nearest neighbor density ratio estimation with model selection',fontsize=16)

plt.subplot(1,2,1)
plt.title('Training and test data with density ratio estimates')
plt.scatter(x_ev[:,0],x_ev[:,1],c=weights,s=100,alpha=0.4,label='evaluation sample',cmap=cmap)
plt.scatter(x_tr[:,0],x_tr[:,1],c=sns.color_palette()[0],label='training sample')
plt.scatter(x_te[:,0],x_te[:,1],c=sns.color_palette()[1],alpha=0.5,label='test sample')
plt.legend(loc='upper left')
plt.axis('equal')

plt.subplot(1,2,2)
plt.title('Density ratio estimates')
plt.scatter(x_ev[:,0],x_ev[:,1],c=weights,s=100,label='evaluation sample',cmap=cmap)
plt.axis('equal')
plt.colorbar()
plt.show()