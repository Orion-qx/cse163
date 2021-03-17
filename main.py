"""
This is the main file
for functions that will plot
and analyze the raw data
coming from the nursing homes.
"""

import sys
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
            "Total Weighted Health Survey Score",
            "Rating Cycle 1 Total Health Score",
            "Rating Cycle 2 Total Health Score",
            "Rating Cycle 3 Total Health Score"]]
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

#def classify(data):

def get_US_data():
    country = gpd.read_file("geo.json")
    country = country[(country['NAME'] != 'Alaska') & 
                      (country['NAME'] != 'Hawaii') & 
                      (country['NAME'] != "District of Columbia") &
                      (country['NAME'] != 'Puerto Rico')]
    return country


def health_vs_rating(data):

    mask1 = data["Total Weighted Health Survey Score"].isna()
    data = data[~mask1]
    data = data.groupby("Provider State", as_index=False).mean()
    abbreviations = pd.read_csv("csvData.csv")
    country = get_US_data()

    data = data.merge(abbreviations, left_on="Provider State", right_on="Code", how="left")
    data = country.merge(data, left_on="NAME", right_on="State", how="left")

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))

    data.plot(column="Overall Rating", ax=ax1, legend=True, cmap='OrRd')
    data.plot(column="Total Weighted Health Survey Score", ax=ax2, legend=True, cmap='OrRd')

    ax1.set_xlabel("Overall Rating Graph")
    ax2.set_xlabel("Weighted Health Survey Score")

    fig.tight_layout()
    plt.savefig("Health_vs_Overall_Rating.png")


    
def health_geo_graphs(cycle1, cycle2, cycle3):

    test = cycle3["Rating Cycle 3 Total Health Score"] == "."
    cycle3 = cycle3[~test]
    cycle1["Rating Cycle 1 Total Health Score"] = (cycle1["Rating Cycle 1 Total Health Score"].astype(float))
    cycle2["Rating Cycle 2 Total Health Score"] = (cycle2["Rating Cycle 2 Total Health Score"].astype(float))
    cycle3["Rating Cycle 3 Total Health Score"] = (cycle3["Rating Cycle 3 Total Health Score"].astype(float))

    cycle1 = cycle1.groupby("Provider State", as_index=False)["Rating Cycle 1 Total Health Score"].mean()
    cycle2 = cycle2.groupby("Provider State", as_index=False)["Rating Cycle 2 Total Health Score"].mean()
    cycle3 = cycle3.groupby("Provider State", as_index=False)["Rating Cycle 3 Total Health Score"].mean()
    
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 5))
    
    abbreviations = pd.read_csv("csvData.csv")
    country = gpd.read_file("geo.json")
    country = get_US_data()

    cycle1 = cycle1.merge(abbreviations, left_on="Provider State", right_on="Code", how="left")
    cycle2 = cycle2.merge(abbreviations, left_on="Provider State", right_on="Code", how="left")
    cycle3 = cycle3.merge(abbreviations, left_on="Provider State", right_on="Code", how="left")

    cycle1 = country.merge(cycle1, left_on="NAME", right_on="State", how="left")
    cycle2 = country.merge(cycle2, left_on="NAME", right_on="State", how="left")
    cycle3 = country.merge(cycle3, left_on="NAME", right_on="State", how="left")

    ax1.set_xlabel("Cycle 1 health Reports")
    ax2.set_xlabel("Cycle 2 health Reports")
    ax3.set_xlabel("Cycle 3 health Reports")

    cycle1.plot(column="Rating Cycle 1 Total Health Score", ax=ax1, cmap='OrRd')
    cycle2.plot(column="Rating Cycle 2 Total Health Score", ax=ax2, cmap='OrRd')
    cycle3.plot(column="Rating Cycle 3 Total Health Score", ax=ax3, legend=True, cmap='OrRd')

    fig.tight_layout()
    plt.savefig("Health_Ratings_Map.png")
    

def health_analysis(data):
    
    new = data[["Total Weighted Health Survey Score",
                "Rating Cycle 1 Total Health Score",
                "Rating Cycle 2 Total Health Score",
                "Rating Cycle 3 Total Health Score",
                'Overall Rating', "Provider State"]]

    mask1 = new["Total Weighted Health Survey Score"].isna()
    mask11 = new["Overall Rating"].isna()
    cycle1 = new[["Total Weighted Health Survey Score", 
                  "Rating Cycle 1 Total Health Score",
                  'Overall Rating', "Provider State"]]
    cycle1 = cycle1[~mask1 & ~mask11]

    mask2 = new["Total Weighted Health Survey Score"].isna()
    mask21 = new["Overall Rating"].isna()
    cycle2 = new[["Total Weighted Health Survey Score", 
                  "Rating Cycle 2 Total Health Score",
                  'Overall Rating', "Provider State"]]
    cycle2 = cycle2[~mask2 & ~mask21]

    mask3 = new["Total Weighted Health Survey Score"].isna()
    mask31 = new["Overall Rating"].isna()
    cycle3 = new[["Total Weighted Health Survey Score", 
                  "Rating Cycle 3 Total Health Score",
                  'Overall Rating', "Provider State"]]
    cycle3 = cycle3[~mask3 & ~mask31]

    health_geo_graphs(cycle1, cycle2, cycle3)
    health_vs_rating(new[["Total Weighted Health Survey Score",
                'Overall Rating', "Provider State"]])
    

def capacity_analysis(data):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    new = data[['Average Number of Residents per Day',
                'Number of Certified Beds',
                'Overall Rating']]

    mask1 = new['Number of Certified Beds'].isna()
    mask2 = new["Average Number of Residents per Day"].isna()
    mask3 = new["Overall Rating"].isna()

    rnew = new[~mask1 & ~mask2 & ~mask3]

    rnew["Percentage Filled"] = rnew["Average Number of Residents per Day"] / rnew['Number of Certified Beds']

    sns.kdeplot(
        data=rnew, x='Number of Certified Beds',
        y="Overall Rating",
        palette="crest",
        log_scale=1.2, ax=ax[0]
        )

    sns.scatterplot(
        data=rnew, y="Overall Rating",
        x="Percentage Filled",
        palette="crest",
        ax=ax[1], 
        )

    ax[0].title.set_text("Ratings vs Number of Certified Beds")
    ax[1].title.set_text("Ratings vs Percentage of Beds Filled")
    fig.tight_layout()
    plt.savefig("sp1.png")


def plot_staffing_hours(info):

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sns.kdeplot(
        data=info, x="Reported Total Nurse Staffing Hours per Resident per Day",
        y="Average Number of Residents per Day",
        hue="Overall Rating",
        fill=True, common_norm=False, palette="crest",
        alpha=.5,
        log_scale=1.2, ax=ax[0]
        )
    

    sns.kdeplot(
        data=info, x="Reported Total Nurse Staffing Hours per Resident per Day",
        y="Overall Rating",
        palette="crest",
        log_scale=1.2, ax=ax[1]
        )

    sns.kdeplot(
        data=info, y="Overall Rating",
        x="Average Number of Residents per Day",
        palette="crest",
        log_scale=1.2, ax=ax[2]
        )
    ax[0].title.set_text("Average Residents Per Day vs Nurse Staffing Hours Per Day")
    ax[1].title.set_text("Ratings vs Nurse Staffing Hours Per Day")
    ax[2].title.set_text("Ratings vs Average Residents Per Day")
    fig.tight_layout()
    plt.savefig("sp.png")

def staffing_hours_data(data):
    
    new = data[['Average Number of Residents per Day',
       'Reported Total Nurse Staffing Hours per Resident per Day',
       'Overall Rating']]

    mask1 = new['Reported Total Nurse Staffing Hours per Resident per Day'].isna()
    mask2 = new["Average Number of Residents per Day"].isna()
    mask3 = new["Overall Rating"].isna()

    rnew = new[~mask1 & ~mask2 & ~mask3]

    plot_staffing_hours(rnew)

def rating_analysis(data):

    data = data[["Provider State", "Overall Rating"]]
    stuff = data.groupby("Provider State", as_index=False)["Overall Rating"].mean()

    #sns.catplot(x='Provider State', y='Overall Rating', data=stuff, aspect=15/5, kind="bar")
    
    plt.savefig("ratings.png")

def provider_vs_rating(data):
    """
    This function takes data
    about the nursing homes
    and makes calculations about
    the different nursing homes
    and graphs them.
    """
    fig, [ax1, ax2] = plt.subplots(2)
    data = data[["Ownership adjusted", "Overall Rating"]]
    grouped = data.groupby("Ownership adjusted", as_index=False)["Overall Rating"].mean()
    sns.barplot(x='Ownership adjusted', y='Overall Rating', data=grouped, ax=ax2)

    box_data = data
    ax = sns.boxplot(x="Ownership adjusted", y="Overall Rating", data=box_data, ax=ax1)
    ax.set(xticklabels=[])
    ax.set(xlabel=None)
    ax.set_title(("Average Overall Ratings for Different Nursing Home Types"))
    fig.savefig("ratings.png")

def provider_vs_rating_geo(data_org):
    """
    This function takes the data about
    the nursing homes and uses the geo
    data to graph the overall ratings
    of the nursing homes based off
    of their location and the type
    of bursing home they are.
    """

    data = data_org[["Provider State", "Ownership adjusted", "Overall Rating", 'Reported Total Nurse Staffing Hours per Resident per Day']]
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 5))
    
    abbreviations = pd.read_csv("csvData.csv")
    country = get_US_data()

    data = data.merge(abbreviations, left_on="Provider State", right_on="Code", how="left")

    country = country.merge(data, left_on="NAME", right_on="State", how="left")
    is_profit = country["Ownership adjusted"] == "For profit"
    profit = country[is_profit]
    non_profit = country["Ownership adjusted"] == "Non profit"
    nprofit = country[non_profit]
    government = country["Ownership adjusted"] == "Government"
    govt = country[government]

    print(profit['Reported Total Nurse Staffing Hours per Resident per Day'].mean())
    print(nprofit['Reported Total Nurse Staffing Hours per Resident per Day'].mean())
    print(govt['Reported Total Nurse Staffing Hours per Resident per Day'].mean())

    ax1.set_xlabel("For profit")
    ax2.set_xlabel("Non profit")
    ax3.set_xlabel("Government")

    ax2.set_title("Overall Ratings for Different Nursing Home Types")
    
    profit.plot(column="Overall Rating", ax=ax1, cmap='OrRd')
    nprofit.plot(column="Overall Rating", ax=ax2, cmap='OrRd')
    govt.plot(column="Overall Rating", ax=ax3, legend=True, cmap='OrRd')

    fig.tight_layout()
    plt.savefig("US_map_ratings.png")


def plot_maps(data, shape):
    print(shape["STATEFP"].unique())


def main():
    data = col_selection()
    data = data_cleaning(data)
    #shape = gpd.read_file('practice/cb_2018_us_state_500k.shp')

    # split_data(data)
    # col_plots(data)
    # model = DecisionTreeClassifier()
    # model.fit(X_train, y_train)
    # label_predictions = model.predict(features)

    #health_analysis(data)
    #staffing_hours_data(data)
    #capacity_analysis(data)
    #rating_analysis(data)
    #provider_vs_rating(data)
    provider_vs_rating_geo(data)

    # plot_maps(data, shape)


if __name__ == '__main__':
    main()