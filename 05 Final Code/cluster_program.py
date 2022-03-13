#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
 
# create a new SparkSession and connect to MongoDB database & collection
spark = SparkSession     .builder     .appName("YouTube Trending Videos Analysis and Prediction")     .config("spark.mongodb.input.uri", "mongodb+srv://gp15:MSBD5003gp15@cluster0.qfnff.mongodb.net/Database0.US_pre")     .getOrCreate()

spark # check if sparksession created successfully


# In[3]:


df = spark.read.format("mongo").option("spark.mongodb.input.uri", "mongodb+srv://gp15:MSBD5003gp15@cluster0.qfnff.mongodb.net/Database0.US_pre").load()
df1 = df.drop("_id")
df1.show()


# In[4]:


df2 = df1.select('category_id','tags','views').withColumnRenamed("views","label")
df2.cache()


# In[8]:


from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import *
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator


# In[9]:


tokenizer = RegexTokenizer(inputCol="tags", outputCol="words", pattern="\|")
tokenized = tokenizer.transform(df2)
tokenized.cache()


# In[10]:


word = tokenized.withColumn("exp", F.explode('words')).select('exp')


# In[11]:


fq = word.groupby('exp').count()
fq = fq.filter(col('count') > 10).collect()
fq = [i['exp'] for i in fq]


# In[12]:


final = tokenized.rdd.map(lambda x: (x.category_id, [i for i in x.words if i in fq], x.label))
final = spark.createDataFrame(final).toDF('id', 'words','label')
final = final.rdd.map(lambda x: (x.id, x.words if (len(x.words)>0) else ['[none]'] , x.label))
final = spark.createDataFrame(final).toDF('id', 'words','label')


# In[13]:


final = final.withColumn('id_list', F.array(final.id))
final = final.withColumn('merged', concat(final.id_list, final.words))


# In[14]:


hashingTF = HashingTF(inputCol='merged', outputCol="features", numFeatures = 2048)
hashed = hashingTF.transform(final)
hashed.show()


# In[15]:


trainingData, testData = hashed.randomSplit([0.8, 0.2])

rf = RandomForestRegressor(featuresCol="features", maxDepth=8)

model = rf.fit(trainingData)

predictionsDf = model.transform(testData)
predictionsDf.show()


# In[16]:


rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
rf_evaluator.evaluate(predictionsDf)


# In[ ]:




