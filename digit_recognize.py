from sklearn.externals import joblib
from sklearn import datasets
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

dataset = fetch_openml('mnist_784', version=1, return_X_y=False)

features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
             block_norm='L1', visualise=None)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()
clf.fit(hog_features, labels)
joblib.dump(clf, "digits_cls.pkl", compress=3)
