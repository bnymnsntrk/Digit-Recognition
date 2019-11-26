from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

dataset = datasets.fetch_mldata("MNIST Original")

features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')


