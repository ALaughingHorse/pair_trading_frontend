import streamlit as st
import time
import numpy as np
import pickle
import plotly.express as px
from streamlit import session_state as ss
import requests

# Define the URL for the POST request
# url = "http://localhost:8000/mlapi-predict"

# # Define the header for the request, specifying the content type
# headers = {
#     'Content-Type': 'application/json',
# }

# # Define the payload (data) to be sent in the POST request
# data = {
#     "duration_in_days": 120,d
#     "dollar_amt": 100
# }

# # Make the POST request to the specified URL with the given headers and data
# response = requests.post(url, json=data, headers=headers)

# # Print the response text to see the result of the request
# print(response.text)

st.set_page_config(page_title="Find Pairs", page_icon="📈")
st.markdown("# Find Pairs")
st.sidebar.header("Find Pairs")
# st.write(
#     """Wanna know the performance of our model? Pick a date in history and examine the performance!"""
# )