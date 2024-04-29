# Databricks notebook source
import time
import mlflow
import requests
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

catalog = "music_genie"
schema = "music_genie_db"

# name for the LLM and endpoint that will label text with emotions
labeler_model_name = "music_genie_langchain_model"
labeler_model_serving_endpoint_name = 'music_genie_emotions_model_endpoint'

# name for the embedding model and endpoint that will convert emotions to embeddings
embeddings_model_name = "music_genie_embeddings_model"
embedding_model_serving_endpoint_name = 'music_genie_embeddings_model_endpoint'

embedding_index = "song_lyrics_emotions_embeddings_index"

# COMMAND ----------

print(f'''You are using the following Unity Catalog location:\nCatalog: {catalog}\nSchema: {schema}\n\n
      To change the location, update the variables in variables.py.''')

# COMMAND ----------

def get_latest_model_version(model_name):
  '''A helper function to grab the latest model version for our labeler model registered in UC.'''
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

# COMMAND ----------

# def create_endpoint(model_serving_endpoint_name, instance, headers, my_json):
#   #get endpoint status
#   endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
#   url = f"{endpoint_url}/{model_serving_endpoint_name}"
#   r = requests.get(url, headers=headers)
#   if "RESOURCE_DOES_NOT_EXIST" in r.text:  
#     print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
#     re = requests.post(endpoint_url, headers=headers, json=my_json)
#   else:
#     new_model_version = (my_json['config'])['served_models'][0]['model_version']
#     print("This endpoint existed previously! We are updating it to a new config with new model version: ", new_model_version)
#     # update config
#     url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
#     re = requests.put(url, headers=headers, json=my_json['config']) 
#     # wait till new config file in place
#     import time,json
#     #get endpoint status
#     url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
#     retry = True
#     total_wait = 0
#     while retry:
#       r = requests.get(url, headers=headers)
#       assert r.status_code == 200, f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
#       endpoint = json.loads(r.text)
#       if "pending_config" in endpoint.keys():
#         seconds = 10
#         print("New config still pending")
#         if total_wait < 6000:
#           #if less the 10 mins waiting, keep waiting
#           print(f"Wait for {seconds} seconds")
#           print(f"Total waiting time so far: {total_wait} seconds")
#           time.sleep(10)
#           total_wait += seconds
#         else:
#           print(f"Stopping,  waited for {total_wait} seconds")
#           retry = False  
#       else:
#         print("New config in place now!")
#         retry = False
#   assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"

# COMMAND ----------

# endpoint helper function
def wait_for_endpoint(model_serving_endpoint_name, instance, headers):
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    while True:
        url =  f"{endpoint_url}/{model_serving_endpoint_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"
 
        status = response.json().get("state", {}).get("ready", {})
        #print("status",status)
        if status == "READY": print(status); print("-"*80); return
        else: print(f"Endpoint not ready ({status}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds
        
# api_url = mlflow.utils.databricks_utils.get_webapp_url()
 
# wait_for_endpoint()
 
# # Give the system just a couple extra seconds to transition
# time.sleep(30)

# COMMAND ----------

import time
def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

# index helper functions
def index_exists(vsc, endpoint_name, index_full_name):
    indexes = vsc.list_indexes(endpoint_name).get("vector_indexes", list())
    return any(index_full_name == index.get("name") for index in indexes)
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 20 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")
