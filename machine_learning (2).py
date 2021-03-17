import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
sns.set()


def col_selection(data):
    """
    Data preparation. Only keep columns we are interested in.
    """
    df = data

    data = df[['Provider State', 'Provider Zip Code', 'Ownership Type',
               'Number of Certified Beds',
               'Average Number of Residents per Day',
               'Date First Approved to Provide Medicare and Medicaid Services',
               'Overall Rating',
               'Reported Total Nurse Staffing Hours per Resident per Day']]
    return data


def data_cleaning(data):
    """
    Remove all NaN in the column "Overall Rating".
    Report the number of NA's in the dataset.
    """
    print("NA summary:")
    print(data.isna().sum())
    data = data.dropna()
    print(data.isna().sum())
    data = categorical_features_adj(data)
    data = continuous_features_adj(data)
    return data


def categorical_features_adj(data):
    """
    Manipulate the categorical features
    """
    # clean ownership types
    ownership_adj = data.apply(lambda x:
                               x["Ownership Type"].split(" - ")[0],
                               axis=1)
    data["Ownership adjusted"] = ownership_adj
    sns.countplot(x='Ownership adjusted', data=data,
                  order=sorted(data['Ownership adjusted'].unique()))
    plt.title("Distribution of Ownership")
    plt.savefig('distribution_of_ownership.png')
    # extract and classify years
    year_name = 'Date First Approved to Provide Medicare and Medicaid Services'
    year_col = data.apply(lambda x: x[year_name].split("-")[0], axis=1)
    data["year"] = year_col
    data["year"] = data.apply(lambda x: classify_years(x["year"]), axis=1)
    data["region"] = data.apply(lambda x:
                                classify_regions(x["Provider State"]), axis=1)
    return data


def classify_regions(abbr):
    """
    Divide the states into 4 regions
    """
    west = ["WA", "OR", "CA", "MT", "ID", "NV", "UT",
            "AZ", "WY", "CO", "NM", "HI", "AK"]
    midwest = ["ND", "SD", "NE", "KS", "MN", "IA",
               "MO", "WI", "IL", "IN", "MI", "OH"]
    south = ["OK", "TX", "AR", "LA", "KY", "TN", "MS",
             "AL", "WV", "VA", "MD", "DE", "NC", "SC", "GA", "FL"]
    if abbr in west:
        return "west"
    elif abbr in midwest:
        return "midwest"
    elif abbr in south:
        return "south"
    else:
        return "northeast"


def classify_years(input_year):
    """
    Divide the years into 4 categories
    """
    input_year = int(input_year)
    if input_year < 1980:
        return "< 1980"
    elif input_year < 1993:
        return "1980-1993"
    elif input_year < 2006:
        return "1993-2006"
    else:
        return ">= 2006"


def continuous_features_adj(data):
    """
    Print out the distribution of the data and
    choose transformation and make it into normal distribution
    """
    beds = 'Number of Certified Beds'
    data[beds] = data.apply(lambda x: np.log(x[beds]), axis=1)
    residents = 'Average Number of Residents per Day'
    data[residents] = data.apply(lambda x: np.log(x[residents]), axis=1)
    hrs = 'Reported Total Nurse Staffing Hours per Resident per Day'
    data[hrs] = data.apply(lambda x: np.log(x[hrs]), axis=1)
    return data


def col_plots_and_ml(data):
    """
    This part contains the correlation plots
    Exploration of machine learning and hyperparameter
    also happens in this part.
    We do the testing by splitting the data into test sets
    and training sets and validate the results based on the
    graph we have.
    """
    data['avg residents'] = data['Average Number of Residents per Day']
    hrs = 'Reported Total Nurse Staffing Hours per Resident per Day'
    data['staffing hrs'] = data[hrs]
    data['beds number'] = data['Number of Certified Beds']
    data['Ownership'] = data['Ownership adjusted']
    data = data[['region', 'Ownership', 'beds number',
                 'avg residents', 'year',
                 'Overall Rating', 'staffing hrs']]
    print(data.head())
    sns.countplot(x='Overall Rating', data=data,
                  order=sorted(data['Overall Rating'].unique()))
    plt.title("Distribution of Overall Rating")
    plt.savefig('distribution_of_overall_rating.png')
    features = data.loc[:, data.columns != 'Overall Rating']
    features = pd.get_dummies(features)
    f, ax = plt.subplots(figsize=(3, 3))
    sns.pairplot(data, hue='Ownership', diag_kind="kde",
                 kind="scatter", palette="husl")
    plt.savefig('corr_between_features_and_label.png')
    f, ax = plt.subplots(figsize=(10, 10))
    corr = features.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.title("Correlation between features")
    plt.xticks(rotation=-25)
    plt.savefig('corr_between_features.png')
    labels = data['Overall Rating']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=2)
    accuracies = []
    for i in range(1, 40):
        model = DecisionTreeClassifier(max_depth=i, random_state=0)
        model.fit(features_train, labels_train)
        train_pred = model.predict(features_train)
        train_acc = accuracy_score(labels_train, train_pred)
        test_pred = model.predict(features_test)
        test_acc = accuracy_score(labels_test, test_pred)
        accuracies.append({'max depth': i, 'train accuracy': train_acc,
                           'test accuracy': test_acc})
    accuracies = pd.DataFrame(accuracies)
    plot_accuracies(accuracies, 'train accuracy', 'Train')
    plot_accuracies(accuracies, 'test accuracy', 'Test')


def plot_accuracies(accuracies, column, name):
    """
    function scrapped from lessons' code
    Parameters:
        * accuracies: A DataFrame show the
        * train/test accuracy for various max_depths
        * column: Which column to plot (e.g., 'train accuracy')
        * name: The display name for this column (e.g., 'Train')
    """
    sns.relplot(kind='line', x='max depth', y=column, data=accuracies)
    plt.title(f'{name} Accuracy as Max Depth Changes')
    plt.xlabel('Max Depth')
    plt.ylabel(f'{name} Accuracy')
    plt.ylim(0, 1)
    plt.savefig(f'{name} Accuracy as Max Depth Changes.png')


def main():
    """
    main method to run all functions above
    """
    data = pd.read_csv("Data.csv", encoding='ISO-8859-1')
    data = col_selection(data)
    data = data_cleaning(data)
    col_plots_and_ml(data)


if __name__ == '__main__':
    main()