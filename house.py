import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import streamlit as st

model = pk.load(open("forest.pkl", "rb"))

st.header("House Price Predictor Machine Learning Model")

data =  pd.read_csv("housing.csv")

st.text("""
        This web app alllows a user to predict the price of a house.
        """)


longitude = st.selectbox("**Longitude**", data['longitude'].unique())
latitude = st.selectbox("**Latitude**", data['latitude'].unique())
housing_median_age = st.selectbox("**Select the house median age**", data["housing_median_age"].unique())
total_rooms = st.selectbox("**No of rooms**", data["total_rooms"].unique())
total_bedrooms = st.selectbox("**No of bedrooms**", data["total_bedrooms"].unique())
population = st.selectbox("**No of people**", data["population"].unique())
households = st.selectbox("**No of households**", data["households"].unique())
median_income = st.selectbox("**Income of a person**", data["median_income"].unique())
inland = st.selectbox("**Income of a person**", data["bedroom_ratio"].unique())
house_location = st.selectbox("**Location of the house**", data["ocean_proximity"].unique())

if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,house_location]],
    columns=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity'])

    house_price = model.predict(input_data_model)

    rounded_price = round(house_price[0])
    st.markdown('**HOUSE PRICE IS GOING TO BE KSH: '+ str(rounded_price) + '**')