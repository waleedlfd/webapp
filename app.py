import streamlit as st
import streamlit_authenticator as stauth
import yaml
from PIL import Image
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from millify import millify
from millify import prettify




import streamlit as st
st.set_page_config(
        page_title="LFD APP",
        layout="wide",
        initial_sidebar_state="expanded",

    )

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#     hide_streamlit_style = """
#             <style>
#             footer {visibility: hidden;}
#             MainMenu {visibility: hidden;}
#             </style>
#             """
#     st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# def add_logo(logo_path, width, height):
#     logo = Image.open(logo_path)
#     modified_logo = logo.resize((width, height,))
#     return modified_logo

# my_logo = add_logo(logo_path="static\\logo_new.png", width=220, height=60)
# st.sidebar.image(my_logo)


st.title('LFD - Credit Scoring Application')

data = pd.read_csv('cust_score.csv')
data_trans = pd.read_csv('cust_trans.csv')
data_col = pd.read_csv('cust_col.csv')
data_traj = pd.read_csv('trajectory2.csv')

cus_filter= pd.read_csv('customer_filter.csv')
data = data.merge(cus_filter, on='Customer_ID',  how='inner')
data_col = data_col.merge(cus_filter, on='Customer_ID', how='inner')
data_trans = data_trans.merge(cus_filter, on='Customer_ID', how='inner')

data['Limit Amount'] = np.round(data['Limit Amount'])
data_trans['Net Cashflow Predicted'] = np.round(data_trans['Net Cashflow Predicted'])

st.subheader('Please upload an excel file with customer identification numbers')
with st.form(key='settings'):
    cus_filter = st.file_uploader(label="Upload CSV", type={"csv", "txt"}, label_visibility='hidden')
    submit_button = st.form_submit_button(label='Submit', use_container_width=True)


if cus_filter is not None:

    st.cache_data()
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(data)
    with st.expander('Table View'):
        st.dataframe(data)
        st.download_button(
        label="Download Predictions as Excel File",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv')


    with st.expander('Dashboard Pane'):
        l, c, r, r2 = st.columns(4)
        l.metric(label='Aggregated Probability of Default', value= np.round(data['Probability of Default'].mean(),2) )
        c.metric(label='Aggregated Credit Score', value= np.round(data['Credit Score'].mean(),2))
        r.metric(label='Aggregated Predicted Net Cash-flow', value= prettify(np.round(data_trans['Net Cashflow Predicted'].mean())))
        r2.metric(label='Aggregated Predicted Limit', value= prettify(np.round(data['Limit Amount'].mean(),2)))
                
        
        left, _, middle, _,  right = st.columns((3, 0.1, 3, 0.1, 5))
        with left:
            counts = data['Risk Category'].value_counts()
            colors = {'High': 'rgba(255, 0, 0, 0.8)',   'Medium': 'rgba(255, 255, 0, 0.8)',  'Low': 'rgba(0, 255, 0, 0.8)'}
            fig1 = px.pie(counts, values=counts.values, names=counts.index, color=counts.index, color_discrete_map=colors, title='Risk Category Breakdown')
            fig1.update_traces(textposition='outside', textinfo='percent+label+value')
            fig1.update_layout(showlegend=False, font=dict(size=12))
            st.plotly_chart(fig1,  use_container_width=True)

        with middle:
            df = data[['On Time', 'DPD 30', 'DPD 60',"DPD 90" ]]
            grouped = df.groupby(df.index)
            means = grouped.agg('mean')
            means.index.name = 'Sample'
            means = pd.melt(means, value_vars=['On Time', 'DPD 30', 'DPD 60',"DPD 90"], var_name='Late Payment', value_name='Probability')
            means = means.groupby('Late Payment')['Probability'].mean().reset_index()

            fig = px.bar(means, x='Probability', y='Late Payment', color='Late Payment', title='Aggregated Payment Status Counts', orientation='h')
            fig.update_layout(showlegend=False, font=dict(size=12))
            fig.update_layout(xaxis_title='Proportion of Customers', yaxis_title='Payment Status', showlegend=False, font=dict(size=12))          
            st.plotly_chart(fig, style={'border': '1px solid white'})

        with right:
            col = data_col.groupby('Collateral Type')['Collateral Value'].sum().reset_index()
            fig = go.Figure(data=[go.Pie(labels=data_col['Collateral Type'], values=data_col['Collateral Value'], hole=0.6, textinfo='label+value')])
            fig.update_traces(textposition='outside', textinfo='percent+label+value')
            fig.update_layout(showlegend=False, font=dict(size=12), title='Collateral Type by Value')
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        data_trans['Date'] = pd.to_datetime(data_trans['Date'])
        data_trans['year'] = data_trans['Date'].dt.year
        data_trans['month'] = data_trans['Date'].dt.month
        data_trans = data_trans.groupby(['year','month'])[['Net Cashflow Predicted', 'Net Cashflow Actual']].mean().reset_index()
        data_trans['day'] = 1
        data_trans['Date'] = pd.to_datetime(data_trans[['year', 'month','day']])

        actual_df = data_trans[data_trans["Date"] <= "2023-05-31"]
        predicted_df = data_trans[(data_trans["Date"] >= "2023-05-01") & (data_trans["Date"] <= "2023-08-31")]

        fig = px.line(actual_df, x="Date", y="Net Cashflow Actual", labels={"Net Cashflow Actual": "Net Cashflow"})
        fig.add_scatter(x=predicted_df["Date"], y=predicted_df["Net Cashflow Predicted"], name="Predicted", mode="lines", line_color="red")
        
        fig.update_layout(title="Actual and Predicted Net Cashflows Over Time", xaxis_title="Date", yaxis_title="Net Cashflow")
        fig.update_traces(name="Actual", selector=dict(name="Net Cashflow Actual"))
        fig.update_traces(name="Predicted", selector=dict(name="Net Cashflow Predicted"))
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        left, _, right = st.columns((4, 0.2, 4))
        with left:
            df_city = data.groupby('City')['Customer_ID'].count().reset_index().rename(columns={'Customer_ID': 'count'})
            city_fig = go.Figure(data=[go.Bar( x=df_city['City'], y=df_city['count'])])
            city_fig .update_layout(showlegend=False, font=dict(size=12), title='City Distribution')
            st.plotly_chart(city_fig, use_container_width=True)

        with right:
            df_edu = data.groupby('Education Level')['Customer_ID'].count().reset_index().rename(columns={'Customer_ID': 'count'})
            edu_fig = go.Figure(data=[go.Bar( x=df_edu['Education Level'], y=df_edu['count'])])
            edu_fig.update_layout(showlegend=False, font=dict(size=12), title='Customer Education Distribution')
            st.plotly_chart(edu_fig, use_container_width=True)

            average_age = data['Age'].mean()
            fig = px.histogram(data, x='Age', nbins=10)
            fig.add_shape(type="line", x0=average_age, x1=average_age, y0=0, y1=50, line=dict(color="red", width=3, dash='dot'))
            fig.update_layout(showlegend=False, font=dict(size=12), title='Age Distribution')
            st.plotly_chart(fig, use_container_width=True)

        with left:
            df_bus = data.groupby('Business')['Customer_ID'].count().reset_index().rename(columns={'Customer_ID': 'count'})
            bus_fig = go.Figure(data=[go.Bar( x=df_bus['Business'], y=df_bus['count'])])
            bus_fig.update_layout(showlegend=False, font=dict(size=12), title='Industry Distribution')
            st.plotly_chart(bus_fig, use_container_width=True)