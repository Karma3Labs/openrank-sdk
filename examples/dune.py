from typing import List, TypedDict
import time

import json
import urllib3
import logging
import csv

GET_NEIGHBORS_URL = 'https://scq1uugv7g.execute-api.us-west-2.amazonaws.com/prod'
DUNE_API_KEY = 'your-api-key'

def get_token_transfer_neighbors(
  addresses: List[str],
  hops: int,
  lim_num_neighbors: int
):
  req = {
    "addresses": addresses,
    "chain_id": 10,
    "lim_num_neighbors":lim_num_neighbors,
    "hops": hops,
  }

  start_time = time.perf_counter()
  try:
    http = urllib3.PoolManager()
    encoded_data = json.dumps(req).encode('utf-8')

    response = http.request('POST', f"{GET_NEIGHBORS_URL}/get_neighbors",
                headers={
                        'Accept': 'application/json',
                        'Content-Type': 'application/json',
                        },
                body=encoded_data,
                timeout=30 * 1000,
                )
    resp_dict = json.loads(response.data.decode('utf-8'))

    if response.status != 200:
      logging.error(f"Server error: {response.status}:{resp_dict} {resp_dict}")
      raise {
        "statusCode": response.status,
        "body": str(resp_dict)
    }

    return resp_dict
  except Exception as e:
    logging.error('error while sending a request to get-p2p-neighbors', e)
  logging.debug(f"get-p2p-neighbors took {time.perf_counter() - start_time} secs ")

example_address = '0xf68d2bfcecd7895bba05a7451dd09a1749026454'
lim_num_neighbors = 200
hops = 2
neighbors = get_token_transfer_neighbors([example_address], hops, lim_num_neighbors)
neighbors = [{ 'i': n['address'], 'v': n['score'] } for n in neighbors]

from openrank_sdk import EigenTrust
sdk = EigenTrust()

dune_table_name = f'openrank_op_token_txfr_{example_address}'
export_path = f'./examples/{dune_table_name}.csv'
description = 'This is a list of addresses with their OpenRank scores based on the token transfer to address for this given users. hops={hops}, lim_num_neighbors={lim_num_neighbors}'
sdk.export_scores_to_csv(neighbors, export_path)
sdk.export_csv_to_dune(export_path, dune_table_name, description, False, DUNE_API_KEY)
