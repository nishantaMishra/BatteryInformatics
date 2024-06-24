"""
This will refresh the data. Includes curve fitting with scatter plot
Working as intended Monday 15 January 2024 12:38:59 PM IST
"""

import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
import seaborn as sns
import readline
import os
import time
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        print("File not found. Make sure the file path is correct.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def input_with_completion(prompt):
    def complete(text, state):
        options = [path for path in glob.glob(text + '*')]
        if state < len(options):
            return options[state] + ' '
        else:
            return None

    readline.set_completer_delims('\t')
    readline.parse_and_bind("tab: complete")

    return input(prompt)


def check_missing_values(df, x_col, y_cols):
    warnings = []

    if not (0 <= x_col < df.shape[1]):
        warnings.append(f"{colored('Caution!','yellow' )} x-axis column number {x_col} is out of range.")

    for y_col in y_cols:
        if not (0 <= y_col < df.shape[1]):
            warnings.append(f"{colored('Caution!','yellow' )} y-axis column number {y_col} is out of range.")

    for warning in warnings:
        print(warning)

    return warnings

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

def draw_scatter_plot(df, x_col, y_cols):
    warnings = check_missing_values(df, x_col, y_cols)
    if warnings:
        return

    x_values = df.iloc[:, x_col]

    plt.figure()

    for y_col in y_cols:
        y_values = df.iloc[:, y_col]
        plt.scatter(x_values, y_values, label=df.columns[y_col])

        curve_fit_option = input("Do you want curve fitting? (Press Enter to skip, or type 'y' for yes): ")

        if curve_fit_option.lower() == 'y':
            degree = int(input("Enter the degree of polynomial to fit (e.g., 1, 2, 3, ...): "))

            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(np.array(x_values).reshape(-1, 1), y_values)

            x_curve = np.linspace(min(x_values), max(x_values), 100)
            y_curve = model.predict(x_curve.reshape(-1, 1))

            plt.plot(x_curve, y_curve, label=f'Curve Fit (Degree {degree})', linestyle='--')

    plt.xlabel(df.columns[x_col])
    plt.ylabel("Values")
    plt.legend()
    plt.title("Scatter Plot with Curve Fitting")
    plt.show()

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

def draw_histogram(df, y_cols):
    plt.figure()
    for y_col in y_cols:
        if not (0 <= y_col < df.shape[1]):
            print(f"{colored('Caution!','yellow' )} Histogram column number {y_col} is out of range.")
            return

        data = df.iloc[:, y_col]
        if data.dtype == bool:
            data = data.astype(int)

        plt.hist(data, bins=10, alpha=0.5, label=df.columns[y_col])
    plt.xlabel(f"{df.columns[y_col]} Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram")
    plt.show()


def draw_pie_chart(df, y_col):
    try:
        plt.figure()
        col_name = df.columns[y_col]
        counts = df.iloc[:, y_col].value_counts()
        angles = 360 * counts / counts.sum()  #
        plt.pie(counts, labels=counts.index, autopct='%.2f')
        plt.title(f"Pie Chart for {col_name}")

        angle_cutoff = 20  #
        if (angles <= angle_cutoff).all():
            print(f"{colored('Notice','red' )}: Profusion of data.")
            print("A Pie chart may not be the most informative visualization for this data.")

        plt.show()

    except IndexError:
        print(f"{colored('Caution!','yellow' )} Pie chart column number {y_col} is out of range.")

def draw_pair_plot(df, columns):
    try:
        columns = columns[:10]

        diag_kind = input("Enter diag_kind (e.g., 'auto', 'hist', 'kde'): ").strip()
        kind = input("Enter kind (e.g., 'scatter', 'reg', 'hex'): ").strip()

        diag_kind = diag_kind if diag_kind else 'auto'
        kind = kind if kind else 'scatter'

        column_numbers = [col_alpha_to_num(col) if not col.isnumeric() else int(col) for col in columns]

        if any(col < 0 or col >= df.shape[1] for col in column_numbers):
            print(f"{colored('Caution!','yellow' )} One or more columns are out of range.")
            return

        invalid_columns = [col for col in column_numbers if col not in range(df.shape[1])]
        if invalid_columns:
            print(f"{colored('Caution!','yellow' )} Column(s) {invalid_columns} not found in the CSV file.")
            return

        sns.pairplot(df.iloc[:, column_numbers], diag_kind=diag_kind, kind=kind)
        plt.show()

    except ValueError as ve:
        print(f"{colored('Caution!','yellow' )} Error in creating pair plot: {ve}")

# Function to draw a Correlation Matrix (Heatmap)
def draw_correlation_matrix(df, columns_range):
    try:
        start_col, end_col = parse_columns_range(columns_range)
        
        if not (0 <= start_col < df.shape[1]) or not (0 <= end_col < df.shape[1]):
            print(f"{colored('Caution!','yellow' )} One or more columns are out of range.")
            return

        invalid_columns = [col for col in range(start_col, end_col + 1) if col not in range(df.shape[1])]
        if invalid_columns:
            print(f"{colored('Caution!','yellow' )} Column(s) {invalid_columns} not found in the CSV file.")
            return

        correlation_matrix = df.iloc[:, start_col:end_col + 1].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix (Heatmap)")
        plt.show()

    except ValueError as ve:
        print(f"{colored('Caution!','yellow' )} Error in creating correlation matrix: {ve}")


def parse_columns_range(columns_range):
    start_col, end_col = map(lambda x: int(x) if x.isnumeric() else col_alpha_to_num(x), columns_range.split(','))
    return start_col, end_col

def col_alpha_to_num(alpha):
    result = 0
    for char in alpha:
        result = result * 26 + ord(char) - ord('A') + 1
    return result - 1


def main():
    csv_file = input_with_completion("Enter the path to your CSV file: ")
    df = load_data(csv_file)
    last_modified_time = os.path.getmtime(csv_file)

    if df is not None:
        while True:
            # Check if the file has been modified
            current_modified_time = os.path.getmtime(csv_file)
            if current_modified_time > last_modified_time:
                df = load_data(csv_file)
                last_modified_time = current_modified_time

            print("Choose a plot type:")
            print("1: Line graph")
            print("2: Scatter Plot")
            print("3: Bar graph")
            print("4: Histogram")
            print("5: Pie chart")
            print("6: Pair plot")
            print("7: Correlation Matrix (Heatmap)")
            print("8: Quit")

            choice = input("Enter your choice: ")

            if choice == '1':
                x_col_input = input("Enter the x-axis column (number or alphabetical notation): ")
                x_col = int(x_col_input) if x_col_input.isnumeric() else col_alpha_to_num(x_col_input)
                y_cols_input = input("Enter the y-axis column(s) (comma-separated): ")
                y_cols = [int(y) if y.isnumeric() else col_alpha_to_num(y) for y in y_cols_input.split(',')]
                draw_line_graph(df, x_col, y_cols)
            elif choice == '2':
                x_col_input = input("Enter the x-axis column (number or alphabetical notation): ")
                x_col = int(x_col_input) if x_col_input.isnumeric() else col_alpha_to_num(x_col_input)
                y_cols_input = input("Enter the y-axis column(s) (comma-separated): ")
                y_cols = [int(y) if y.isnumeric() else col_alpha_to_num(y) for y in y_cols_input.split(',')]
                draw_scatter_plot(df, x_col, y_cols)
            elif choice == '3':
                x_col_input = input("Enter the x-axis column (number or alphabetical notation): ")
                x_col = int(x_col_input) if x_col_input.isnumeric() else col_alpha_to_num(x_col_input)
                y_cols_input = input("Enter the y-axis column(s) (comma-separated): ")
                y_cols = [int(y) if y.isnumeric() else col_alpha_to_num(y) for y in y_cols_input.split(',')]
                draw_bar_graph(df, x_col, y_cols)
            elif choice == '4':
                y_cols_input = input("Enter the column(s) for the histogram (comma-separated): ")
                y_cols = [int(y) if y.isnumeric() else col_alpha_to_num(y) for y in y_cols_input.split(',')]
                draw_histogram(df, y_cols)
            elif choice == '5':
                y_col_input = input("Enter the column for the Pie chart (number or alphabetical notation): ")
                y_col = [int(y_col_input) if y_col_input.isnumeric() else col_alpha_to_num(y_col_input)]
                draw_pie_chart(df, y_col)
            elif choice == '6':
                columns_input = input("Enter columns for the pair plot (comma-separated): ")
                columns = [col.strip() for col in columns_input.split(',')]
                draw_pair_plot(df, columns)
            elif choice == '7':
                columns_range_input = input("Enter the range of columns for the correlation matrix (e.g., 'A, C' or '1, 3'): ")
                draw_correlation_matrix(df, columns_range_input)
            elif choice == '8':    
                break
            else:
                print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
