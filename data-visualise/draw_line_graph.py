import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored
from utils import check_missing_values

# Function to draw a Line graph with sorted x_col data
def draw_line_graph(df, x_col, y_cols):
    warnings = check_missing_values(df, x_col, y_cols)
    if warnings:
        return

    sorted_indices = df.iloc[:, x_col].argsort()
    sorted_x = df.iloc[sorted_indices, x_col]

    plt.figure()
    for y_col in y_cols:
        sorted_y = df.iloc[sorted_indices, y_col]
        plt.plot(sorted_x, sorted_y, label=df.columns[y_col])
    plt.xlabel(df.columns[x_col])
    plt.ylabel("Values")
    plt.legend()
    plt.title("Line Graph (Sorted by x_col)")
    plt.show()
