import streamlit as st
import time
import numpy as np
import pickle
import pandas as pd
import json
import plotly.express as px
from streamlit import session_state as ss
import requests

# Define the URL for the POST request
url = "http://3.101.78.185:8000/mlapi-predict"

# Define the header for the request, specifying the content type
headers = {
    'Content-Type': 'application/json',
}

def ConvertResponseToTable(sinput):
	df = pd.DataFrame(sinput, columns = ['Pair_1', 'Pair_2', 'Probability'])
	df.sort_values(by='Probability', ascending=False)
	return df

st.set_page_config(page_title="Find Pairs", page_icon="📈")
st.markdown("# Find Pairs")
st.sidebar.header("Find Pairs")

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Find Pairs using the 🔮 tool', on_click=click_button)

requested_pairs = st.slider('How many pairs would you like?', 1, 100, 1, 1, on_change=click_button)

if st.session_state.clicked:
	input_pairs = requested_pairs
	# Define the payload (data) to be sent in the POST request
	data = {
		"requested_pairs": requested_pairs,
	    "duration_in_days": 120,
	    "dollar_amt": 100
	}

	# Send the request
	response = requests.post(url, json=data, headers=headers)

	# Read as json
	input_data = json.loads(response.text)

	st.table(ConvertResponseToTable(input_data['predictions']))
	st.session_state.clicked = False
	