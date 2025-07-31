import requests
import pandas as pd 
from pandas import Timedelta
import datetime
import numpy as np


def download_data(symbol):
  ACCESS_TOKEN = 'Amt21OY7tkBqJ1ylAP4mlQreVx45'
  headers = {
    'Authorization': 'Bearer {}'.format(ACCESS_TOKEN),
    'Accept': 'application/json'
  }
  start_date = '2020-01-01'
  # end_date = '2023-09-01'



  # Prepare the request
  url = 'https://api.tradier.com/v1/markets/history'
  headers = headers
  params = {
      'symbol': symbol,
      'interval': 'daily',
      'start': start_date,
    #   'end': end_date,
      'session_filter': 'open'
  }

  # Send the request
  response = requests.get(url, headers=headers, params=params)

  # Check for a successful response
  if response.status_code == 200:
      # Convert the response to a DataFrame
      data = response.json()
      df = pd.DataFrame(data['history']['day'])
      # print(df)
  else:
      print(f"Error: {response.status_code}")
  return df

