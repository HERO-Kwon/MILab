import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans,AgglomerativeClustering

# Load data
word_map = np.load('wordvec_map.npy')
# yolo result w/ category
yolo_cat = pd.read_csv('yolo1_1201.csv')

# filter data
yolo_model = yolo_cat
#yolo_model = yolo_cat[yolo_cat.supercategory!='accessory']
#yolo_model = yolo_cat[yolo_cat.supercategory.isin(['vehicle','animal','indoor','outdoor'])]
wordvec_model = word_map[yolo_model.index]
yolo_model.head()

#dimension reduction using PCA
pca = PCA(n_components=2,whiten=True)
pca.fit(word_map)

#plotting PCA
from matplotlib.pyplot import cm
X_pca = pca.transform(wordvec_model)
supercat = set(yolo_model.supercategory)
target_ids = range(len(supercat))
color=cm.tab20(target_ids)
#color=cm.tab20(np.linspace(0,1,13))
from matplotlib import pyplot as plt
plt.figure()
for i, c, label in zip(target_ids, color, supercat):
    plt.scatter(X_pca[yolo_model.supercategory==label, 0], X_pca[yolo_model.supercategory==label, 1],c=c, label=label)
plt.legend()
plt.show()
# K means
from sklearn.cluster import KMeans
# create model and prediction
model = KMeans(n_clusters=len(set(yolo_model.supercategory)),algorithm='auto',random_state=10)
#model = AgglomerativeClustering(n_clusters=len(set(yolo_model.supercategory)))
model.fit(wordvec_model)
predict = pd.DataFrame(model.predict(wordvec_model))
#predict = pd.DataFrame(model.labels_)
predict.columns=['predict']
predict.index = yolo_model.index

ct = pd.crosstab(yolo_model['supercategory'],predict['predict'])
print (ct)

#plotting PCA
from matplotlib.pyplot import cm
X_pca = pca.transform(wordvec_model)
supercat = set(predict['predict'])
target_ids = range(len(supercat))
color=cm.tab20(target_ids)
#color=cm.tab20(np.linspace(0,1,13))
from matplotlib import pyplot as plt
plt.figure()
for i, c, label in zip(target_ids, color, supercat):
    plt.scatter(X_pca[predict['predict']==label, 0], X_pca[predict['predict']==label, 1],c=c, label=label)
plt.legend()
plt.show()