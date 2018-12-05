import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
word_map = np.load('wordvec_map.npy')
# yolo result w/ category
yolo_cat = pd.read_csv('yolo_cat.csv')

#dimension reduction using PCA
pca = PCA(n_components=2,whiten=True)
pca.fit(word_map)

#plotting
from matplotlib.pyplot import cm
X_pca = pca.transform(word_map)
supercat = set(yolo_cat.supercategory)
target_ids = range(len(supercat))

color=cm.tab20(np.linspace(0,1,13))
from matplotlib import pyplot as plt
plt.figure()
for i, c, label in zip(target_ids, color, supercat):
    plt.scatter(X_pca[yolo_cat.supercategory==label, 0], X_pca[yolo_cat.supercategory==label, 1],c=c, label=label)
plt.legend()
plt.show()

#kmeans classifier
kmeans = KMeans(n_clusters=13,random_state=5).fit(word_map)

#plotting
from matplotlib.pyplot import cm
X_pca = pca.transform(word_map)
#categories = yolo_cat.supercategory
categories = kmeans.labels_
cat = set(categories)
target_ids = range(len(supercat))

color=cm.tab20(np.linspace(0,1,13))
from matplotlib import pyplot as plt
plt.figure()
for i, c, label in zip(target_ids, color, cat):
    plt.scatter(X_pca[categories==label, 0], X_pca[categories==label, 1],c=c, label=label)
plt.legend()
plt.show()