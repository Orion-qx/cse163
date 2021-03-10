"""
This is the main file
for functions that will plot
and analyze the raw data
coming from the nursing homes.
"""

import sys

import numpy
import pandas as pd
import matplotlib.pyplot as plt

import requests
import seaborn as sns
import sklearn
import geopandas as gpd

sns.set()

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

    data = data[["Ownership Type", "Overall Rating"]]
    grouped = data.groupby("Ownership Type", as_index=False)["Overall Rating"].mean()
    is_profit = grouped["Ownership Type"].str.contains("For profit")
    is_profit_value = float(grouped[is_profit].mean())
    non_profit = grouped["Ownership Type"].str.contains("Non profit")
    non_profit_value = float(grouped[non_profit].mean())
    gov_profit = grouped["Ownership Type"].str.contains("Government")
    gov_profit_value = float(grouped[gov_profit].mean())

    stuff = pd.DataFrame([{"Ownership Type": 'For Profit', 'Rating': is_profit_value},
                          {"Ownership Type": 'Government', 'Rating': gov_profit_value},
                          {"Ownership Type": 'Non Profit', 'Rating': non_profit_value}])
    print(stuff.head)

    sns.barplot(x='Ownership Type', y='Rating', data=stuff)
    
    plt.savefig("ratings.png")


def main():

    data = pd.read_csv("Data.csv", encoding='ISO-8859-1')
    #staffing_hours_data(data)
    #rating_analysis(data)
    provider_vs_rating(data)


if __name__ == '__main__':
    main()