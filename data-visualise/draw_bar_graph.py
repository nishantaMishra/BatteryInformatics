import matplotlib.pyplot as plt
from termcolor import colored
from utils import check_missing_values

# Function to draw a Bar graph
def draw_bar_graph(df, x_col, y_cols):
    warnings = check_missing_values(df, x_col, y_cols)
    if warnings:
        return

    plt.figure()
    for y_col in y_cols:
        plt.bar(df.iloc[:, x_col], df.iloc[:, y_col], label=df.columns[y_col])
    plt.xlabel(df.columns[x_col])
    plt.ylabel("Values")
    plt.legend()
    plt.title("Bar Graph")
    plt.show()
