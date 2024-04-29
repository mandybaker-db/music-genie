# Databricks notebook source
# MAGIC %md
# MAGIC ### Embeddings / Create Embedding Model Serving Endpoint
# MAGIC #### Using all-distilroberta-v1
# MAGIC
# MAGIC Cluster using runtime 
# MAGIC
# MAGIC
# MAGIC ***Upgraded cluster to 14.3 LTS 

# COMMAND ----------

# MAGIC %%capture
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch mlflow==2.9.1 -U sentence-transformers requests "urllib3<2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./variables

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to manage our own embeddings for this project, although you can also choose to use embeddings managed for you by Databricks. We're going to use an embedding model from HuggingFace.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

# chose an embedding model based on HuggingFace suggestions, as well as input length and purpose
# https://huggingface.co/sentence-transformers/all-distilroberta-v1
embeddings_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's load our song dataset again, and test out our embedding model against some of the labeled emotions.

# COMMAND ----------

# load the song lyrics and corresponding emotions label
labeled_df = spark.sql("select * from music_genie.music_genie_db.spotify_songs_with_labeled_emotions limit 3")

# COMMAND ----------

# test the embedding model
song_emotions = labeled_df.select('llm_labeled_emotions').first()[0]
print("Emotions: ", song_emotions, "\n")

embeddings = embeddings_model.encode(song_emotions)
print(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's register this embedding model... 

# COMMAND ----------

from transformers import pipeline
from mlflow.models import infer_signature
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

# use Unity Catalog to store the model
mlflow.set_registry_uri('databricks-uc') 

# define the path to them model in UC
# we've set our catalog, schema, and embeddings_model_name variables in the .variables file to reference here
registered_model_name = f'{catalog}.{schema}.{embeddings_model_name}'

# Compute input/output schema
signature = infer_signature(song_emotions, embeddings_model.encode(song_emotions))

with mlflow.start_run():

  # using the transformers flavor of MLflow models
  # if you've already registered the model, this step will version up the existing model
  model_info = mlflow.sentence_transformers.log_model(
    embeddings_model,
    artifact_path="all_distilroberta_v1_embedding_model",
    signature=signature,
    input_example=song_emotions,
    registered_model_name=registered_model_name)

# COMMAND ----------


# grabbing the model's latest version and setting a model alias so that we can quickly access the right version later when setting up our embedding model endpoint
def get_latest_model_version(model_name):
  '''A helper function to grab the latest model version for our labeler model registered in UC.'''
  model_version_infos = MlflowClient().search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

model_alias = "music_genie_embedding_model"

MlflowClient().set_registered_model_alias(registered_model_name, "music_genie_embedding_model", get_latest_model_version(registered_model_name)) 

# COMMAND ----------

# MAGIC %md
# MAGIC ... and save it to a model serving endpoint.
# MAGIC
# MAGIC *** explain why we need an endpoint for the embedding model, because we'll need to calculate the embeddings for user input on the fly for vector search to work (same as masking model endpoint) (e.g. get similarities back quickly)
# MAGIC
# MAGIC *** only need to run this once

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
 
# With the token, you can create our authorization header for our subsequent REST calls
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }
 
# Next you need an endpoint at which to execute your request which you can get from the notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
 
# This object comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
 
# Lastly, extract the Databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the embedding model endpoint.

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import requests

embedding_model_serving_endpoint_name = 'music_genie_embeddings_model_endpoint'
client = get_deploy_client("databricks")

try:
    endpoint = client.create_endpoint(
        name=embedding_model_serving_endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": f'{registered_model_name}',
                    "entity_version": get_latest_model_version(model_name=registered_model_name),
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True
                }
            ]
        }
    )
    print(f"Endpoint {embedding_model_serving_endpoint_name} is being created. This may take several minutes.")
except requests.exceptions.HTTPError as err:
    error_message = err.response.text
    if "Endpoint with name" in error_message and embedding_model_serving_endpoint_name in error_message and "already exists" in error_message:
        print(f"Endpoint {embedding_model_serving_endpoint_name} already exists.")

# COMMAND ----------

import time, mlflow

instance = tags["browserHostName"]
 
def wait_for_endpoint():
    '''Gets and returns the serving endpoint status so that we don't proceed with following steps until the endpoint is ready.'''
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    while True:
        url =  f"{endpoint_url}/{embedding_model_serving_endpoint_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"
        status = response.json().get("state", {}).get("ready", {})
        if status == "READY": print(status); print("-"*80); return
        else: print(f"Endpoint {embedding_model_serving_endpoint_name} not ready ({status}), waiting 60 seconds."); time.sleep(60) # Wait 60 seconds
        
api_url = mlflow.utils.databricks_utils.get_webapp_url()
 
wait_for_endpoint()
 
# Give the system just a couple extra seconds to transition
time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC Test out the endpoint.

# COMMAND ----------

import requests
 
def score_model(data_json: dict):
    url =  f"https://{instance}/serving-endpoints/{embedding_model_serving_endpoint_name}/invocations"
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()['predictions']

payload_json = {
  "dataframe_split": {
    "data": [
        ["love, happiness, joy, sincerity, peace"]
      ], 
    } 
  }

print(score_model(payload_json))

# COMMAND ----------

# MAGIC %md
# MAGIC Awesome! We have an embeddings model that will calculate the embeddings for both the emotions we've already labeled our songs with, as well as an endpoint to host the model so that we can compute the embeddings for the emotions that we'll determine from user input.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Embeddings and save them to a Delta table.

# COMMAND ----------

import mlflow.deployments
from pyspark.sql.functions import pandas_udf, udf
import pandas as pd

@udf("array<float>")
def get_embeddings(input_string):

  payload_json = {
  "dataframe_split": {
    "data": [
        input_string
      ], 
    } 
  }

  url = f"https://{instance}/serving-endpoints/{embedding_model_serving_endpoint_name}/invocations"
  response = requests.request(method="POST", headers=headers, url=url, json=payload_json)
  if response.status_code != 200:
      raise Exception(f"Request failed with status {response.status_code}, {response.text}")
  return response.json()['predictions'][0]

# COMMAND ----------

labeled_df = spark.sql("select * from music_genie.music_genie_db.spotify_songs_with_labeled_emotions")
embedded_df = labeled_df.select("artist", "song", "link", "llm_labeled_emotions", get_embeddings('llm_labeled_emotions').alias("emotions_embeddings"))

display(embedded_df)

# COMMAND ----------

# save the table
embedded_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.spotify_song_emotions_embeddings")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- enabling Change Data Feed in order to create an index
# MAGIC ALTER TABLE music_genie.music_genie_db.spotify_song_emotions_embeddings SET TBLPROPERTIES (delta.enableChangeDataFeed=true);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Self-Managed Vector Search Index.
# MAGIC Databricks Vector Search has several options for indexes, including managed and self-managed. Since we have opted to use a lightweight embedding model, we are going to manage the embeddings ourselves. This means there will be additional work each time we update our dataset with more songs.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# using a shared workspace endpoint; one endpoint can host up to 20 indexes
vector_search_endpoint_name = 'one-env-shared-endpoint-4'

# check if the endpoint has already been created and create if needed
if vector_search_endpoint_name not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
    vsc.create_endpoint(name=vector_search_endpoint_name, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, vector_search_endpoint_name)
print(f"Endpoint named {vector_search_endpoint_name} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# the table we want to index; for self-managed embeddings, this is the table that includes the pre-computed embeddings
source_table_full_name = f"{catalog}.{schema}.spotify_song_emotions_embeddings"

# the table we use to store the vector search index
vs_index_full_name = f"{catalog}.{schema}.spotify_song_emotions_embeddings_index"

# we are using a triggered pipeline; this is more cost effective, but it means that you will have to manually update the index (index.sync()) when new songs and embeddings are added to the source table
if not index_exists(vsc, vector_search_endpoint_name, vs_index_full_name):
  print(f"Creating index {vs_index_full_name} on endpoint {vector_search_endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=vector_search_endpoint_name,
    index_name=vs_index_full_name,
    source_table_name=source_table_full_name,
    pipeline_type="TRIGGERED",
    primary_key="link",
    embedding_dimension=768, # It's important to match your embedding model size here
    embedding_vector_column="emotions_embeddings"
  )

# wait for vector search index to be ready and embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vector_search_endpoint_name, vs_index_full_name)
print(f'Index {vs_index_full_name} on table {source_table_full_name} is ready.')

# COMMAND ----------

# MAGIC %md
# MAGIC Our Vector Search Index is ready! Move on to the final notebook to use vector search against user input and get song recommendations.
