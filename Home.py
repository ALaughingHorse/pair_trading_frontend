import streamlit as st
from PIL import Image

icon = Image.open('images/ppf_logo.png')
# Set the page layout to wide
st.set_page_config(
    page_title="Parallel Portfolios",
    page_icon=icon
)


st.write("# Welcome to Parallel Portfolios!")
st.sidebar.image('images/ppf_logo.png')
st.sidebar.success("Select a tab above")

st.markdown(
    """
With Parallel Portfolios we want to democratize data science techniques to help retail investors generate investment returns the same way nuanced portfolio managers do. 
Parallel Portfolios empowers individual investors to confidently engage in pair trading. Unlike traditional approaches that limit pair selection to within the same industry, our unified methodology expands opportunities for users to benefit from pair trading across diverse industries.

Parallel Portfolios strategically invests users' funds across identified profitable pairs. Users gain transparency into their investment performance by selecting specific date ranges. Additionally, users have the flexibility to customize the refresh period for their investments, with our industry-recommended default set at 60 days.

Our platform offers users insights through simulations, allowing them to forecast the performance of their stocks. Moreover, users have access to analytics highlighting the most traded, most profitable, and most loss-inducing pairs, enabling informed decision-making and deeper understanding of their investment strategies.
"""
)

st.write("Check out our official webpage [Parallel Portfolios](https://www.ischool.berkeley.edu/projects/2024/parallel-portfolios)")

st.write("Check out our official [Github](https://github.com/ALaughingHorse/pair_trading/tree/main)")

