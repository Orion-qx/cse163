"""
This is the main file
for functions that will plot
and analyze the raw data
coming from the nursing homes.
"""

import sys
import json
import numpy
import pandas as pd
import seaborn as sns
import geopandas as gpd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

sns.set()
# X_train = None
# y_train = None
# X_test = None
# y_test = None
# features = None
# labels = None


def col_selection():
    """
    Data preparation. Only keep columns we are interested in.
    """
    df = pd.read_csv("NH_ProviderInfo_Feb2021.csv", encoding="ISO-8859-1")

    data = df[['Provider State', 'Provider Zip Code', 'Ownership Type', 'Number of Certified Beds',
            'Average Number of Residents per Day', 'Date First Approved to Provide Medicare and Medicaid Services', 
            'Overall Rating', 'Reported Total Nurse Staffing Hours per Resident per Day', 
            'Total Weighted Health Survey Score']]
    return data


def data_cleaning(data):
    """
    Remove all NaN in the column "Overall Rating".
    Report the number of NA's in the dataset.
    """
    print(data.isna().sum())  # number NA's for each column
    num_of_nan = data["Overall Rating"].isna().sum()
    print(f"There are {num_of_nan} rows that have NaN in the 'Overall Rating' column")
    data = data.dropna()

    # clean ownership types
    # print(data['Ownership Type'].unique())
    data["Ownership adjusted"] = data["Ownership Type"].astype(str).str.split(" - ").str[0]
    return data


# def split_data(data):
#     features = data.loc[:, data.columns != 'Overall Rating']
#     labels = data['Overall Rating']
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
# if using states as feature, need one-hot-encoding


def col_plots(data):
    """
    Some data visualization for the data
    """
    sns.countplot(x = 'Overall Rating', data = data, order=sorted(data['Overall Rating'].unique()))
    plt.title("Distribution of Overall Rating")
    plt.savefig('distribution_of_overall_rating.png')

    sns.countplot(x='Ownership adjusted', data=data, order=sorted(data['Ownership adjusted'].unique()))
    plt.title("Ownership")
    #plt.xticks(rotation=-45)
    plt.savefig('ownership.png')

    # fig = px.histogram(data, x = "Ownership Type")
    # fig.write_image('ownership_px.png')

    ##### -- works but have no labels --- #####
    features = data.loc[:, data.columns != 'Overall Rating']
    # plt.matshow(features.corr())
    # plt.title("Correlation of Features")
    # plt.savefig('correlation_of_features')
    ##########

    ##### -- works on jupyter notebook -- #####
    # corr = features.corr()
    # corr.style.background_gradient(cmap='coolwarm')
    #####
    

def plot_state_map_avg(data):
    """
    Using a new package plotly, creates an interactive map that plots the average Overall Rating
    for each state. 
    """
    data = data.groupby('Provider State', as_index = False)['Overall Rating'].mean()
    fig = go.Figure(data=go.Choropleth(
        locations=data['Provider State'], # Spatial coordinates
        z = data['Overall Rating'].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Blues',
        colorbar_title = "Overall Rating",
        marker_line_color='white',
        marker_line_width=0.5
    ))

    fig.update_layout(
        title_text = 'Average Overall Rating by State',
        geo_scope='usa', # limite map scope to USA
    )

    fig.show()


def plot_state_map_pct(data):
    """
    Using a new package plotly, creates an interactive map that shows the percent of nursing homes
    with an overall rating 5 for each state. 
    """
    total = data.groupby('Provider State', as_index = False)['Overall Rating'].sum()

    five = data[data['Overall Rating'] == 5]
    five = five.groupby('Provider State', as_index = False)['Overall Rating'].sum()
    five['Pct Rating 5'] = (five['Overall Rating'] / total['Overall Rating']) * 100
    print(five)
    fig = go.Figure(data=go.Choropleth(
        locations=five['Provider State'], # Spatial coordinates
        z = five['Pct Rating 5'].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Magma',
        colorbar_title = "Percent",
        marker_line_color='white',
        marker_line_width=0.5
    ))

    fig.update_layout(
        title_text = 'Percent of Nursing Homes with Rating of 5 by State',
        geo_scope='usa', # limite map scope to USA
    )
    fig.show()


def rating_resident_num_correlation(data): 
    rating = data.groupby('Provider State', as_index = False)['Overall Rating'].mean()
    num_residents = data.groupby('Provider State', as_index = False)['Average Number of Residents per Day'].mean()
    num_residents['Average Rating'] = rating['Overall Rating']

    fig = px.scatter(num_residents, x='Average Number of Residents per Day', y='Average Rating', 
                     trendline="ols")

    fig.update_layout(
        title_text = 'Average Overall Rating vs. Avg Number of Residents',
    )

    fig.show()



def main():
    data = col_selection()
    data = data_cleaning(data)

    # Extra geospatial data 
    
    state = gpd.read_file('practice/tl_2020_us_state.shp')
    zip_code = gpd.read_file('practice/cb_2019_us_zcta510_500k.shp')
    states = gpd.read_file('practice/gz_2010_us_040_00_5m.json')
    shape = gpd.read_file('practice/geojson-counties-fips.json')

   
    # split_data(data)
    col_plots(data)
    # model = DecisionTreeClassifier()
    # model.fit(X_train, y_train)
    # label_predictions = model.predict(features)

    plot_state_map_avg(data)
    plot_state_map_pct(data)
    rating_resident_num_correlation(data)


if __name__ == '__main__':
    main()