import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
plt.style.use('seaborn')

st.set_page_config(layout='wide')
st.title('Tanzania Water Wells')

@st.cache
def load_data():
    with open('data/clean_training_values.pickle', 'rb') as file:
        data_df = pickle.load(file)
    train_labels = pd.read_csv('data/training_set_labels.csv')
    data_df['Labels'] = train_labels['status_group']
    # Renaming for map function
    data_df = data_df.rename(columns={'imputed_longitude':'lon', 'imputed_latitude':'lat'})
    return data_df

@st.cache
def get_prediction(model_values):
    # Loading Data to allow imputing values that weren't input
    pipe_xgb = joblib.load("data/model.pkl") # Load "model.pkl"
    model_columns = joblib.load("data/model_columns.pkl") # Load "model_columns.pkl"
    with open('data/clean_training_values.pickle', 'rb') as file:
        data_df = pickle.load(file)

    # Making Label Encoder to reconvert predictions into strings from numbers
    train_labels = pd.read_csv('data/training_set_labels.csv')
    y_label = LabelEncoder()
    y_label.fit(train_labels['status_group'])

    # Making Dataframe contain columns used in saved model
    input_data = pd.DataFrame.from_dict(model_values)
    query = input_data.reindex(columns=model_columns, fill_value=np.nan)

    # Imputing reasonable values for non-input values
    manage = model_values['management'][0]
    query['scheme_management'] = data_df[data_df['management']==manage]['scheme_management'].mode()
    s = model_values['source'][0]
    query['source_class'] = data_df[data_df['source']==s]['source_class'].mode()
    etc = model_values['extraction_type_class'][0]
    query['extraction_type'] = data_df[data_df['extraction_type_class']==etc]['extraction_type'].mode()
    for location_data in ['basin','lga','imputed_latitude', 'imputed_longitude','imputed_population']:
        loc = model_values['ward'][0]
        query[location_data] = data_df[data_df['ward']==loc][location_data].mode()
    for col in query.columns[query.isna().any()].tolist():
        query[col] = query[col].fillna(value=data_df[col].mode())

    # Making Prediction
    prediction = pipe_xgb.predict(query)
    prediction = y_label.inverse_transform(prediction)
    return prediction[0]


data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Done! (using st.cache)")

st.subheader('What Is This?')
st.markdown(
    """
    This is a dashboard for data from the Pump It Up competetion showing the functioning of water pumps in Tanzania along with a live version of a classification model
    that predicts the functioning of hypothetical water pumps from user input.
    For more information on its construction go to https://github.com/nonlocal-lia/pump_it_up_competition.
    """
    )

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data.head())

st.subheader('Options for Display Map')
interesting_variables = ['funder', 'installer', 'management_group',
 'permit', 'water_quality', 'quantity', 'source',
 'waterpoint_type', 'extraction_type_class', 'public_meeting']
variable = st.selectbox('Variable to Map:', tuple(interesting_variables))
option = st.selectbox('Option to Map:', tuple(data[variable].value_counts().index.tolist()))
st.write('You selected:', option)
statuses = ['all', 'functional', 'non functional', 'functional needs repair']
status = st.selectbox('Pump Status:', tuple(statuses))
filtered_data = data[data[variable] == option]
if status != 'all':
    filtered_data = filtered_data[filtered_data['Labels']==status]

st.subheader('Map of {x} pumps with {y}: {z}'.format(z=option, y=variable, x=status))
st.map(filtered_data)

st.subheader('Predictive Model')
region = st.selectbox('Region:', tuple(data.region.value_counts().index.tolist()))
region_district = st.selectbox('Region_District:',
 tuple(data[data['region']==region]['region_district'].value_counts().index.tolist()))
ward = st.selectbox('Ward:',
 tuple(data[data['region_district']==region_district]['ward'].value_counts().index.tolist()))
elevation = st.slider('Elevation:',
 int(data[data['ward']==ward]['imputed_gps_height'].min()),
 int(data[data['ward']==ward]['imputed_gps_height'].max()))
funder = st.selectbox('Funder:', tuple(data.funder.value_counts()[:10].index.tolist()))
installer = st.selectbox('Installer:', tuple(data.installer.value_counts()[:10].index.tolist()))
management_group = st.selectbox('Management Group:', tuple(data.management_group.value_counts().index.tolist()))
management = st.selectbox('Manager:',
 tuple(data[data['management_group']==management_group]['management'].value_counts()[:10].index.tolist()))
water_quality = st.selectbox('Water Quality:', tuple(data.water_quality.value_counts().index.tolist()))
water_quantity = st.selectbox('Water Quantity:', tuple(data.quantity.value_counts().index.tolist()))
source = st.selectbox('Water Source:', tuple(data.source.value_counts().index.tolist()))
season = st.selectbox('Season:', tuple(data.season.value_counts().index.tolist()))
pump_age = st.slider('Pump Age (years):', 0, 50)
extraction_type_class = st.selectbox('Extraction Type:', tuple(data.extraction_type_class.value_counts().index.tolist()))
waterpoint_type = st.selectbox('Waterpoint Type:', tuple(data.waterpoint_type.value_counts().index.tolist()))

model_values = {"funder": [funder],
 "installer": [installer],
 "region": [region],
 "region_district": [region_district],
 "ward": [ward],
 'management_group' : [management_group],
 "management": [management],
 "water_quality": [water_quality],
 "quantity": [water_quantity],
 "source": [source], 
 "season": [season],
 "pump_age": [pump_age],
 "imputed_gps_height": [float(elevation)],
 "waterpoint_type": [waterpoint_type],
 "extraction_type_class": [extraction_type_class]
 }
prediction = get_prediction(model_values)
st.subheader('Prediction:')
st.write(prediction)

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Model constructed by <a href="https://github.com/nonlocal-lia/pump_it_up_competition" >Lia Elwonger </a> using data from Pump It Up competition</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)