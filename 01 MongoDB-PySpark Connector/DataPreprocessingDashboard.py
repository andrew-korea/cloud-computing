#!/usr/bin/env python
# coding: utf-8

# # MongoDB PySpark Connector
# Official guide: https://docs.mongodb.com/spark-connector/current/python-api/

# In[2]:


from pyspark.sql import SparkSession

# stop the SparkSession created automatically (by the time the notebook is running, cannot change much in that session's configuration)
# [Ref]https://www.edureka.co/community/5268/how-to-change-the-spark-session-configuration-in-pyspark
spark.sparkContext.stop() 
# create a new SparkSession and connect to MongoDB database & collection
spark = SparkSession     .builder     .appName("YouTube Trending Videos Analysis and Prediction")     .config("spark.mongodb.input.uri", "mongodb+srv://gp15:MSBD5003gp15@cluster0.3ygtx.mongodb.net/Database0.US")     .config("spark.mongodb.output.uri", "mongodb+srv://gp15:MSBD5003gp15@cluster0.qfnff.mongodb.net/Database0.US_preprocessed")     .getOrCreate()

spark # check if sparksession created successfully


# ## Read from MongoDB

# In[3]:


df = spark.read.format("mongo").load()


# In[4]:


df.printSchema()


# In[5]:


df.select('view_count','likes','dislikes','comment_count').describe().show()


# In[6]:


print("Total number of videos (duplicate count for same video on different date)", df.count())


# In[7]:


# check number of rows with null entry the 10 columns of interest
# 查询某列为null的行数
from pyspark.sql.functions import isnull

cols = ['trending_date','title','channelTitle',
        'categoryId', 'publishedAt','tags','view_count',
        'likes','dislikes','comment_count']
for col in cols:
    print(df.filter(isnull('%s'%col)).count())


# In[8]:


df.dropna(thresh=10).count()


# In[9]:


# For considering unique videos only
# remove duplicates for these columns
df_unique = df.dropDuplicates(['video_id'])
print("Total number of unique videos:", df_unique.count())


# ### Read Category_id.json

# In[10]:


df_cat = spark.read.format("mongo").option("uri", "mongodb+srv://gp15:MSBD5003gp15@cluster0.3ygtx.mongodb.net/Database0.US_cat").load()


# In[11]:


from pyspark.sql import functions as F

# Explode Array to Structure
explodej = df_cat.withColumn('Exp_RESULTS',F.explode(F.col('items'))).drop('items')
dfj = explodej.select("Exp_RESULTS.snippet.title",'Exp_RESULTS.id')
dfj=dfj.withColumnRenamed('id','categoryId')
dfj=dfj.withColumnRenamed('title','categoryTitle')     # 将title重命名为category_title
dfj.show(truncate=False)


# In[12]:


df_joined = df.join(dfj, 'categoryId', "left_outer")
# df_.show()
df_final = df_joined.select("trending_date","video_id","title","categoryTitle","channelTitle","channelId","tags","description","publishedAt","view_count","comment_count","dislikes","likes","ratings_disabled","comments_disabled")


# In[ ]:


df_p = df_final.toPandas()
df_p.head()


# In[ ]:





# ## (testing) Write to MongoDB
# ### Notes:
# DataFrameWriter.mode(saveMode)
# 
# -Specifies the behavior when data or table already exists. Options include:
# 
# - append: Append contents of this DataFrame to existing data.
# 
# - overwrite: Overwrite existing data.
# 
# - error or errorifexists: Throw an exception if data already exists.
# 
# - ignore: Silently ignore this operation if data already exists.
# 
# [Ref] http://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrameWriter.mode.html

# In[ ]:


# people = spark.createDataFrame([("Bilbo Baggins",  60), ("Gandalf", 1000), ("Thorin", 195), ("Balin", 178), ("Kili", 77),
#    ("Dwalin", 169), ("Oin", 167), ("Gloin", 158), ("Fili", 82), ("Bombur", None)], ["name", "age"])


# In[13]:


df_final.write.format("mongo").mode("overwrite").option("database", "Database0").option("collection", "US_preprocessed_test").save()


# In[ ]:




