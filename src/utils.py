import pandas as pd
from math import gcd, floor, log
from sklearn.preprocessing import MinMaxScaler
import pingouin as pg
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.formula.api import gls
from scipy import stats
from itertools import combinations
import numpy as np


def bw_scale(annotations_df, columns=["word", "specificity"], raw=False):
    """
    Calculate specificity scores for words based on annotations and optionally return raw counts.
    
    Parameters:
    annotations_df (pd.DataFrame): A DataFrame containing the annotations with columns for items and their best/worst ratings.
    columns (list): A list of column names for the output DataFrame. Default is ["word", "specificity"].
    raw (bool): If True, return a DataFrame with raw counts of best/worst ratings and total counts for each word. If False, return a DataFrame with words and their calculated specificity scores. Default is False.
    
    Returns:
    pd.DataFrame: A DataFrame containing either the specificity scores or the raw counts for each word, depending on the value of the 'raw' parameter.
    """
    annotations_df = annotations_df.dropna()
    all_items = set()
    count_best = {}
    count_worst = {}
    count = {}

    for i, r in annotations_df.iterrows():

        items = r[["item_1", "item_2", "item_3", "item_4"]].values.tolist()
        for item in items:
            all_items.add(item)
            if not item in count:
                count[item] = 0
            count[item] += 1
            if item not in count_best.keys():
                count_best[item] = 0
            if item not in count_worst.keys():
                count_worst[item] = 0
        best = r["best"]
        worst = r["worst"]
        count_best[best] += 1
        count_worst[worst] += 1

    scores = {id: (count_best[id] - count_worst[id]) / count[id] for id in all_items}

    if raw:
        combined_counts = {
            item: {
                "best": count_best[item],
                "worst": count_worst[item],
                "count": count[item],
                "score": scores[item],
            }
            for item in all_items
        }
        combined_counts = pd.DataFrame(
            [
                {
                    "word": item,
                    "best": counts["best"],
                    "worst": counts["worst"],
                    "count": counts["count"],
                    "score": counts["score"],
                }
                for item, counts in combined_counts.items()
            ]
        )
        return combined_counts.sort_values(by="word")

    else:
        out_df = pd.DataFrame(scores.items(), columns=columns).sort_values(
            by=columns[-1], ascending=False
        )
        return out_df


def make_tuples(instances, tuple_size, repetition_factor):
    """
    Generate tuples of specified size from a list of instances, ensuring that the number of instances is coprime to the tuple size.
    
    Parameters:
    instances (list): A list of instances to be used for generating tuples.
    tuple_size (int): The desired size of each tuple.
    repetition_factor (int): The number of times to repeat the tuple generation process.
    
    Returns:
    dict: A dictionary where keys are tuple IDs and values are lists of instances forming the tuples.
    """

    n = len(instances)

    while gcd(n, tuple_size) != 1:
        instances = instances[:-1]
        n = len(instances)

    tuples = dict()
    tuple_id = 0
    for j in range(repetition_factor):
        for x in range(int(floor(n / tuple_size))):
            t = [
                (x * (tuple_size ** (j + 1)) + (i * (tuple_size**j))) % n
                for i in range(tuple_size)
            ]

            tuples[tuple_id] = [instances[x] for x in t]
            tuple_id += 1

    return tuples


def logfreq(raw_freq):
    """
    Apply a logarithmic transformation to a raw frequency count.
    
    Parameters:
    raw_freq (int): The raw frequency count to be transformed.
    
    Returns:
    float: The logarithmically transformed frequency count, calculated as 1 + log(raw_freq).
    """
    return 1 + log(raw_freq)


def look_at(df, x, y, title=None):
    """
    Visualize the relationship between two variables in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x (str): The column name for the x-axis variable.
    y (str): The column name for the y-axis variable.
    title (str, optional): The title of the plot. If not provided, a default title
                           based on the column names will be used.

    This function performs the following visualizations:
    1. A scatter plot with a lowess regression line to observe the relationship between x and y.
    2. Kernel density estimation (KDE) plots for the distributions of x and y.
    3. A 2D KDE plot to visualize the joint distribution of x and y.
    """

    # scatter plot with lowess regression line
    sns.regplot(
        data=df,
        x=x,
        y=y,
        lowess=True,
        scatter=True,
        line_kws={"color": "orange", "lw": 2},
    )
    plt.show()

    # kde plots for x and y
    sns.kdeplot(df[x], fill=True)
    sns.kdeplot(df[y], fill=True)
    plt.legend([x.upper(), y.upper()])
    # if title is specified, use it; otherwise, create a default title
    if title:
        plt.title(title)
    else:
        plt.title(f"Distribution of {x.upper()} and {y.upper()}")
    plt.xlabel(None)
    plt.show()

    # 2d kde plot for joint distribution of x and y
    sns.displot(df, x=x, y=y, kind="kde", fill=True)
    plt.show()


def scale_columns(df, columns):
    """
    Scale specified columns of a DataFrame to a range between 0 and 1.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be scaled.
    columns (list): A list of column names to scale.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns scaled to the range [0, 1].

    This function uses MinMaxScaler from sklearn.preprocessing to scale the values
    of the specified columns. Each column is scaled independently.
    """

    # scale specified columns to the range [0, 1]
    scaler = MinMaxScaler()
    for col in columns:
        df[col] = scaler.fit_transform(df[[col]])

    return df


def do_stats(df, x, y):
    """
    Perform statistical analysis on two variables in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x (str): The column name for the x-axis variable.
    y (str): The column name for the y-axis variable.

    Returns:
    None

    This function performs the following statistical analyses:
    1. Shapiro-Wilk or Jarque-Bera test for normality on both variables, depending on the sample dimension.
    2. Spearman correlation test if at least one variable is not normally distributed, else
       Pearson correlation test.
    3. Generalized Least Squares (GLS) regression analysis with y as dependent and x as independent variables.
    The results of the correlation tests and regression analysis are printed to the console.
    The function also prints a message indicating the type of correlation test performed
    based on the normality of the variables.
    """
    print(f"Performing statistical analysis between {x.upper()} and {y.upper()}...\n")

    # determine which normality test to use based on the sample dimension
    if len(df) <= 2000:
        normality_test = "Shapiro-Wilk"
    else:
        normality_test = "Jarque-Bera"

    print(f"Performing {normality_test} test for normality...")

    # normality test
    if normality_test == "Shapiro-Wilk":
        # Shapiro-Wilk test for normality
        x_pval = stats.shapiro(df[x]).pvalue
        y_pval = stats.shapiro(df[y]).pvalue
    else:
        # Jarque-Bera test for normality
        x_pval = stats.jarque_bera(df[x]).pvalue
        y_pval = stats.jarque_bera(df[y]).pvalue

    if any([x_pval < 0.05, y_pval < 0.05]):
        # speerman correlation
        method = "spearman"
        print(
            f"Based on the {normality_test} test, at least one variable is not normally distributed.\nExecuting {method.upper()} correlation"
        )
    else:
        # pearson correlation
        print(
            f"Based on the {normality_test} test, both variables are normally distributed. Executing {method.upper()} correlation"
        )
        method = "pearson"
    corr_out = pg.corr(x=df[x], y=df[y], method=method)
    print(corr_out)
    print("\n############################\n")

    # linear regression
    regression_formula = f"{y} ~ {x}"
    print(f"Performing regression analysis with formula: {regression_formula.upper()}")

    # fit a Generalized Least Squares (GLS) model with the regression formula
    reg_model = gls(formula=regression_formula, data=df).fit()

    print(reg_model.summary())
    """
    Perform statistical analysis on two variables in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x (str): The column name for the x-axis variable.
    y (str): The column name for the y-axis variable.

    Returns:
    None

    This function performs the following statistical analyses:
    1. Shapiro-Wilk or Jarque-Bera test for normality on both variables, depending on the sample dimension.
    2. Spearman correlation test if at least one variable is not normally distributed, else
       Pearson correlation test.
    3. Generalized Least Squares (GLS) regression analysis with y as dependent and x as independent variables.
    The results of the correlation tests and regression analysis are printed to the console.
    The function also prints a message indicating the type of correlation test performed
    based on the normality of the variables.
    """
    print(f"Performing statistical analysis between {x.upper()} and {y.upper()}...\n")

    if len(df) <= 2000:
        normality_test = "Shapiro-Wilk"
    else:
        normality_test = "Jarque-Bera"
    print(f"Performing {normality_test} test for normality...")
    # Perform normality test
    if normality_test == "Shapiro-Wilk":
        # Shapiro-Wilk test for normality
        x_pval = stats.shapiro(df[x]).pvalue
        y_pval = stats.shapiro(df[y]).pvalue
    else:
        # Jarque-Bera test for normality
        x_pval = stats.jarque_bera(df[x]).pvalue
        y_pval = stats.jarque_bera(df[y]).pvalue

    if any([x_pval < 0.05, y_pval < 0.05]):
        # Perform Spearman correlation test
        method = "spearman"
        print(
            f"Based on the {normality_test} test, at least one variable is not normally distributed.\nExecuting {method.upper()} correlation"
        )
    else:
        # Perform Pearson correlation test
        print(
            f"Based on the {normality_test} test, both variables are normally distributed. Executing {method.upper()} correlation"
        )
        method = "pearson"
    corr_out = pg.corr(x=df[x], y=df[y], method=method)
    print(corr_out)
    print("\n############################\n")
    # Perform linear regression
    regression_formula = f"{y} ~ {x}"
    print(f"Performing regression analysis with formula: {regression_formula.upper()}")
    # Fit a Generalized Least Squares (GLS) model with the regression formula
    reg_model = gls(formula=regression_formula, data=df).fit()

    print(reg_model.summary())
