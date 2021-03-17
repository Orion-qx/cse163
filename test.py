"""
This is the main file
for functions that will plot
and analyze the raw data
coming from the nursing homes.
"""

import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

sns.set()


def col_selection():
    """
    Data preparation. Only keep columns we are interested in.
    """
    df = pd.read_csv("NH_ProviderInfo_Feb2021.csv", encoding="ISO-8859-1")

    data = df[['Provider State', 'Provider Zip Code', 'Ownership Type',
               'Number of Certified Beds',
               'Average Number of Residents per Day',
               'Date First Approved to Provide Medicare and Medicaid Services',
               'Overall Rating',
               'Reported Total Nurse Staffing Hours per Resident per Day',
               'Total Weighted Health Survey Score']]
    return data


def data_cleaning(data):
    """
    Remove all NaN in the column "Overall Rating".
    Report the number of NA's in the dataset.
    """
    print(data.isna().sum())  # number NA's for each column
    print(data["Overall Rating"].isna().sum())
    data = data.dropna()

    # clean ownership types
    # print(data['Ownership Type'].unique())
    data["Ownership adjusted"] = data["Ownership Type"].astype(str).str \
        .split(" - ").str[0]
    return data


def plot_state_map_avg(data):
    """
    Using a new package plotly, creates an interactive map that plots the
    average Overall Rating for each state.
    """
    data = data.groupby('Provider State', as_index=False)['Overall Rating'] \
        .mean()
    fig = go.Figure(data=go.Choropleth(
        locations=data['Provider State'],  # Spatial coordinates
        z=data['Overall Rating'].astype(float),  # Data to be color-coded
        locationmode='USA-states',  # set locations match entrie in `locations`
        colorscale='Blues',
        colorbar_title="Overall Rating",
        marker_line_color='white',
        marker_line_width=0.5
    ))

    fig.update_layout(
        title_text='Average Overall Rating by State',
        geo_scope='usa',  # limite map scope to USA
    )

    fig.show()


def plot_state_map_pct(data):
    """
    Using a new package plotly, creates an interactive map that shows the
    percent of nursing homes with an overall rating 5 for each state.
    """
    total = data.groupby('Provider State', as_index=False)['Overall Rating'] \
        .sum()

    five = data[data['Overall Rating'] == 5]
    five = five.groupby('Provider State', as_index=False)['Overall Rating'] \
        .sum()
    five['Pct Rating 5'] = (five['Overall Rating']/total['Overall Rating'])*100
    # print(five)
    fig = go.Figure(data=go.Choropleth(
        locations=five['Provider State'],  # Spatial coordinates
        z=five['Pct Rating 5'].astype(float),  # Data to be color-coded
        locationmode='USA-states',
        colorscale='Magma',
        colorbar_title="Percent",
        marker_line_color='white',
        marker_line_width=0.5
    ))

    fig.update_layout(
        title_text='Percent of Nursing Homes with Rating of 5 by State',
        geo_scope='usa',  # limite map scope to USA
    )
    fig.show()


def rating_resident_num_correlation(data):
    """
    Using a new package plotly, creates an scatter plot that show the
    correlation between the average overall rating (thus the quality) vs the
    average number of residents.
    """
    rating = data.groupby('Provider State', as_index=False)['Overall Rating']\
        .mean()
    num_residents = data.groupby('Provider State', as_index=False
                                 )['Average Number of Residents per Day'] \
        .mean()
    num_residents['Average Rating'] = rating['Overall Rating']

    fig = px.scatter(num_residents, x='Average Number of Residents per Day',
                     y='Average Rating', trendline="ols")

    fig.update_layout(
        title_text='Average Overall Rating vs. Avg Number of Residents',
    )

    fig.show()


def main():
    data = col_selection()
    data = data_cleaning(data)

    plot_state_map_avg(data)
    plot_state_map_pct(data)
    rating_resident_num_correlation(data)


if __name__ == '__main__':
    main()
