import streamlit as st
from PIL import Image

icon = Image.open('images/ppf_logo.png')
# Set the page layout to wide
st.set_page_config(
    page_title="Pair Trading Hub",
    page_icon=icon
)


st.write("# Welcome to Pair Trading Hub!")
st.sidebar.image('images/ppf_logo.png')
st.sidebar.success("Select a tab above")

st.markdown(
    """
Pair trading is a market-neutral trading strategy that involves identifying two highly correlated financial instruments (like stocks, ETFs, currencies, commodities) and taking simultaneous opposing positions on them—buying one (the underperformer) and short-selling the other (the overperformer). The rationale behind this strategy is that the relative prices of these instruments will converge over time. The profit is made from the narrowing of the gap, regardless of the market direction. Traders use statistical and quantitative methods to identify pairs with a historical tendency to move together and then trade on the assumption that any divergence in their price relationship is temporary. This strategy aims to capitalize on temporary market inefficiencies and is considered low risk if properly executed, as it's designed to be unaffected by market movements.
"""
)