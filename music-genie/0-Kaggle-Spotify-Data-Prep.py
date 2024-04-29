# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1: Download Data with Kaggle API & Secrets

# COMMAND ----------

# MAGIC %md
# MAGIC Dataset: https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset

# COMMAND ----------

# MAGIC %sh pip install kaggle

# COMMAND ----------

import pandas as pd
import pyspark.pandas as ps
import os

# COMMAND ----------

# MAGIC %sql
# MAGIC -- create catalog catalog_name;
# MAGIC -- create scheme schema_name;
# MAGIC use catalog music_genie;
# MAGIC use schema music_genie_db;
# MAGIC create volume if not exists kaggle_datasets;

# COMMAND ----------

os.environ['KAGGLE_USERNAME'] = dbutils.secrets.get(scope = "music-genie-scope", key = "kaggle-username")
os.environ['KAGGLE_KEY'] = dbutils.secrets.get(scope = "music-genie-scope", key = "kaggle-key")

# COMMAND ----------

# MAGIC %md
# MAGIC Need to explain how to get a Kaggle Key, and options for creating a secret?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the Kaggle API to download a dataset and save to a Volume

# COMMAND ----------

# MAGIC %sh
# MAGIC export KAGGLE_USERNAME=$KAGGLE_USERNAME
# MAGIC export KAGGLE_KEY=$KAGGLE_KEY
# MAGIC
# MAGIC # list out kaggle datasets related to keywords
# MAGIC kaggle datasets list -s spotify-million-song-dataset

# COMMAND ----------

# MAGIC %sh
# MAGIC export KAGGLE_USERNAME=$KAGGLE_USERNAME
# MAGIC export KAGGLE_KEY=$KAGGLE_KEY
# MAGIC
# MAGIC # download a kaggle dataset to ephemeral storage
# MAGIC kaggle datasets download -d notshrirang/spotify-million-song-dataset -p /tmp/kaggle-spotify-song-data

# COMMAND ----------

# list out the downloaded data in ephemeral storage
dbutils.fs.ls('file:/tmp/kaggle-spotify-song-data/spotify-million-song-dataset.zip')

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip -d /tmp/kaggle-spotify-song-data /tmp/kaggle-spotify-song-data/spotify-million-song-dataset.zip

# COMMAND ----------

dbutils.fs.ls('file:/tmp/kaggle-spotify-song-data/')

# COMMAND ----------

# move data from ephemeral storage to volumes
dbutils.fs.mv("file:/tmp/kaggle-spotify-song-data/spotify_millsongdata.csv", "/Volumes/music_genie/music_genie_db/kaggle_datasets/spotify_millsongdata.csv")

# COMMAND ----------

from pyspark.sql.functions import col


# File location
file_location = "/Volumes/music_genie/music_genie_db/kaggle_datasets/spotify_millsongdata.csv"

# # CSV options
file_type = "csv"
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .option("multiline", "true") \
  .option("escape", "\\") \
  .option("escape", '"') \
  .load(file_location)

display(df.filter(col("text").isNotNull()).limit(200))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving Table To Catalog

# COMMAND ----------

# downsampling to about 5000 songs
sampled_df = df.sample(withReplacement=False, fraction=0.1)

# COMMAND ----------

sampled_df.count()

# COMMAND ----------

sampled_df.write.mode("overwrite").saveAsTable("music_genie.music_genie_db.spotify_songs")
