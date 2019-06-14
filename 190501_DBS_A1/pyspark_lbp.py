#1. Py. SP: Import Library
# Import Library
# Python
import random
import os
import numpy as np
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for graph

# SPARK
import pyspark
import findspark # to find location where spark installed
findspark.init()
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

#2. Py.SP: Make Session

# SPARK: Make Session
sc = SparkContext()
spark = SparkSession(sc)

# Set Target Animal
target_animal = 'alligator'


#3. Py.SP: Make LBP Features From Images

# SPARK: read image files in directory and make it to dataframe
img_dir = "D:\\Data\\AnimalsOnTheWeb\\" + target_animal
imgs = spark.read.format("image").load(img_dir)
imgs.printSchema()

# Python: Make Features from Images and save it to CSV file
#!python lbp.py

# SPARK: Read Feature CSV file and make DataFrame
import pyspark.sql.types as typ
res_lbp = spark.read.csv('Res_LBP.csv',header=True)
labels =[
    ('ind',typ.IntegerType()), # index
    ('Animal',typ.StringType()), # Class of animals
    ('File',typ.StringType()), # filename
    ('ID',typ.StringType()), # picture ID
    ('LBP0',typ.FloatType()), # LBP features
    ('LBP1',typ.FloatType()),
    ('LBP2',typ.FloatType()),
    ('LBP3',typ.FloatType()),
    ('LBP4',typ.FloatType()),
    ('LBP5',typ.FloatType()),
    ('LBP6',typ.FloatType()),
    ('LBP7',typ.FloatType()),
    ('LBP8',typ.FloatType()),
    ('LBP9',typ.FloatType()),
]
# Define Schema
schema = typ.StructType([
    typ.StructField(e[0],e[1],False) for e in labels
])

# CSV read
res_lbp = spark.read.csv('Res_LBP.csv',header=True,schema=schema)
# Select Target Animal
target_lbp = res_lbp.where(res_lbp.Animal.isin(target_animal))
target_lbp.printSchema()


target_lbp.head() #show 1st row

target_lbp.show() # show 20 row

#4. Py.SP: Make Ground Truth

# Python: Read .mat file
import scipy.io as sio # Library for .mat files
import re # Library for Regular Expression
file_path = 'D:\\Data\\AnimalsOnTheWeb\\' + target_animal + '\\'
file = 'animaldata_'+ target_animal + '.mat'
# Read from .mat files
data_read = sio.loadmat(os.path.join(file_path,file))

# truth table (1 or 0)
truth_tbl = list(data_read['gt'][0]) 

# get picture ID and save it to 'name' column
truth_nameread = list(data_read['imgnames'][0])
truth_name = [t[0] for t in truth_nameread]
truth_lists = pd.DataFrame({'name': truth_name,'truth': truth_tbl})
truth_lists['name'] = truth_lists['name'].astype('str')
re_picid = re.compile('pic\d+')
truth_lists['ID'] = [re_picid.findall(r)[0] for r in truth_lists['name']]
truth_lists.head()

# SPARK: convert pandas DF to Spark DF
df_truth = spark.createDataFrame(truth_lists)
df_truth.printSchema()

# Cast Truth column to integer
df_truth = df_truth.withColumn('truth_int',df_truth['truth'].cast(typ.IntegerType()))
df_truth.printSchema()

#show 5 row
df_truth.show(5)

#5. SP: join features and Grd Truth dataframe

df_ml = df_truth.join(target_lbp,on='ID')
df_ml.head(5)

# Select columns from dataframe
df_ml1 = df_ml.select([c for c in df_ml.columns if c in ['truth_int','LBP0','LBP1','LBP2','LBP3','LBP4','LBP5','LBP6','LBP7','LBP8','LBP9']])
df_ml1.show(5)

#6. SP: Machine Learning
#6.1. Feature Creator
# make Feature column
import pyspark.ml.feature as ft
labels_feat =[
    ('LBP0',typ.FloatType()),
    ('LBP1',typ.FloatType()),
    ('LBP2',typ.FloatType()),
    ('LBP3',typ.FloatType()),
    ('LBP4',typ.FloatType()),
    ('LBP5',typ.FloatType()),
    ('LBP6',typ.FloatType()),
    ('LBP7',typ.FloatType()),
    ('LBP8',typ.FloatType()),
    ('LBP9',typ.FloatType()),
]
featuresCreator = ft.VectorAssembler(
    inputCols=[col[0] for col in labels_feat[0:]],outputCol='features'
)

#6.2. Make Classification Model
# make model
import pyspark.ml.classification as cl
logistic = cl.LogisticRegression(maxIter=10,regParam=0.01,labelCol='truth_int')

#6.3. Pipeline

# make pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[featuresCreator,logistic])

# Separate training and test data
lbp_train, lbp_test = df_ml1.randomSplit([0.7,0.3],seed=100)
# Train model
model = pipeline.fit(lbp_train)
# Test
test_model = model.transform(lbp_test) # get results on test dataset
test_model.take(1)

#7. SP: Evaluation
# Evaluation
import pyspark.ml.evaluation as ev
evaluator = ev.BinaryClassificationEvaluator(rawPredictionCol='probability',labelCol='truth_int')
print('Area Under ROC: ' + str(evaluator.evaluate(test_model, {evaluator.metricName:'areaUnderROC'})))


#8. SP: K-means Clustering

import pyspark.ml.clustering as clus
kmeans = clus.KMeans(k=10, featuresCol='features')

# make pipeline
pipeline = Pipeline(stages=[featuresCreator,kmeans])

# Get All class data
df_km = res_lbp.select([c for c in df_ml.columns if c in ['LBP0','LBP1','LBP2','LBP3','LBP4','LBP5','LBP6','LBP7','LBP8','LBP9']])
df_km.show(5)

# Separate training and test data
km_train, km_test = df_km.randomSplit([0.7,0.3],seed=100)

model_km = pipeline.fit(km_train)
test_km = model_km.transform(km_test)

# Show Results
test_km.groupBy('prediction').agg({'*':'count','LBP0':'avg','LBP1':'avg'}).collect()

