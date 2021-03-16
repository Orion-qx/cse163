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
#import plotly.express as px
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
    df = pd.read_csv("Data.csv", encoding="ISO-8859-1")

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

#def classify(data):
    

def plot_staffing_hours(info):
    sns.catplot(x='Provider State', y='percent', data=info, aspect=15/5, kind="bar")
    
    plt.savefig("sp.png")

def staffing_hours_data(data):
    Fnew = data
    
    new = Fnew[['Provider State', 'Average Number of Residents per Day',
       'Reported Total Nurse Staffing Hours per Resident per Day',
       'Case-Mix Total Nurse Staffing Hours per Resident per Day',
       'Adjusted Total Nurse Staffing Hours per Resident per Day',
       'Overall Rating']]
    mask1 = new['Reported Total Nurse Staffing Hours per Resident per Day'].isna()
    mask2 = new['Case-Mix Total Nurse Staffing Hours per Resident per Day'].isna()
    mask3 = new['Adjusted Total Nurse Staffing Hours per Resident per Day'].isna()

    snew = new[mask1 & mask2 & mask3]
    snew = snew.groupby("Provider State", as_index=False)["Overall Rating"].count()
    snew = snew.rename(columns={"Overall Rating" : "removed"})

    rnew = new[~mask1 & ~mask2 & ~mask3]
    rnew = rnew.groupby("Provider State", as_index=False)["Overall Rating"].count()
    rnew = rnew.rename(columns={"Overall Rating" : "available"})

    Lnew = rnew.merge(snew, left_on="Provider State", right_on="Provider State")
    Lnew["percent"] = Lnew["removed"] / (Lnew["available"] + Lnew["removed"])

    plot_staffing_hours(Lnew)

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
    fig.savefig("ratings.png")

def provider_vs_rating_geo(data):
    """
    This function takes the data about
    the nusgin homes and uses the geo
    data to graph the overall ratings
    of the nursing homes based off
    of their location and the type
    of bursing home they are.
    """

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(30, 10))
    abbreviations = pd.read_csv("csvData.csv")
    country = gpd.read_file("geo.json")
    country = country[(country['NAME'] != 'Alaska') & 
                      (country['NAME'] != 'Hawaii') & 
                      (country['NAME'] != "District of Columbia") &
                      (country['NAME'] != 'Puerto Rico')]

    data = data.merge(abbreviations, left_on="Provider State", right_on="Code", how="left")

    country = country.merge(data, left_on="NAME", right_on="State", how="left")
    is_profit = country["Ownership adjusted"] == "For profit"
    profit = country[is_profit]
    non_profit = country["Ownership adjusted"] == "Non profit"
    nprofit = country[non_profit]
    government = country["Ownership adjusted"] == "Government"
    govt = country[government]

    ax1.set_xlabel("For profit")
    ax2.set_xlabel("Non profit")
    ax3.set_xlabel("Government")

    profit.plot(column="Overall Rating", legend=True, ax=ax1, cmap='OrRd')
    nprofit.plot(column="Overall Rating", legend=True, ax=ax2, cmap='OrRd')
    govt.plot(column="Overall Rating", legend=True, ax=ax3, cmap='OrRd')
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

    # staffing_hours(data)
    # staffing_hours_data(data)
    # rating_analysis(data)
    #provider_vs_rating(data)
    provider_vs_rating_geo(data)

    # plot_maps(data, shape)


if __name__ == '__main__':
    main()