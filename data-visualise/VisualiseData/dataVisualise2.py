"""
This will refresh the data. Includes curve fitting with scatter plot
Working as intended on Wednesday 18 September 2024 11:36:30 AM IST

Contains on extra feature in correlation matrix function to NOT
label correlation values if number of features is more that 45. 


Known bug: Code will stop while trying to curve fit for scatter plot if data is missing.
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


# Load CSV file into a DataFrame
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

# Function to enable tab-completion for file paths
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

# Function to check and inform about missing values in specified columns
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

#function to draw scatter plot with regression 


def draw_scatter_plot(df, x_col, y_cols):
    # Check for missing values
    warnings = check_missing_values(df, x_col, y_cols)
    if warnings:
        return

    # Select X and Y columns from the DataFrame
    x_values = df.iloc[:, x_col]

    plt.figure()

    for y_col in y_cols:
        y_values = df.iloc[:, y_col]
        plt.scatter(x_values, y_values, label=df.columns[y_col])

        # Ask the user if they want curve fitting
        curve_fit_option = input("Do you want curve fitting? (Press Enter to skip, or type 'y' for yes): ")

        if curve_fit_option.lower() == 'y':
            degree = int(input("Enter the degree of polynomial to fit (e.g., 1, 2, 3, ...): "))

            # Perform curve fitting using polynomial regression
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(np.array(x_values).reshape(-1, 1), y_values)

            # Generate values for the curve
            x_curve = np.linspace(min(x_values), max(x_values), 100)
            y_curve = model.predict(x_curve.reshape(-1, 1))

            # Plot the curve
            plt.plot(x_curve, y_curve, label=f'Curve Fit (Degree {degree})', linestyle='--')

            # Extract polynomial coefficients
            lin_reg = model.named_steps['linearregression']
            poly_features = model.named_steps['polynomialfeatures']
            coeffs = lin_reg.coef_.flatten()
            intercept = lin_reg.intercept_

            # Build and print the polynomial equation
            equation = f"y = {intercept:.4f} "
            for i, coef in enumerate(coeffs[1:], start=1):  # skip first coef, which is intercept-related
                equation += f"+ {coef:.4f}*x^{i} "

            print(f"Polynomial Equation (Degree {degree}): {equation}")

    plt.xlabel(df.columns[x_col])
    plt.ylabel("Values")
    plt.legend()
    plt.title("Scatter Plot with Curve Fitting")
    plt.show()

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

# Function to draw a Histogram
def draw_histogram(df, y_cols):
    plt.figure()
    for y_col in y_cols:
        if not (0 <= y_col < df.shape[1]):
            print(f"{colored('Caution!','yellow' )} Histogram column number {y_col} is out of range.")
            return

        data = df.iloc[:, y_col]
        if data.dtype == bool:
            # Convert boolean values to integers (0 for False, 1 for True)
            data = data.astype(int)

        plt.hist(data, bins=10, alpha=0.5, label=df.columns[y_col])
    plt.xlabel(f"{df.columns[y_col]} Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram")
    plt.show()

# Function to draw a Pie chart with the column name in the title and a suggestion
def draw_pie_chart(df, y_col):
    try:
        plt.figure()
        col_name = df.columns[y_col]
        counts = df.iloc[:, y_col].value_counts()
        angles = 360 * counts / counts.sum()  # Calculate the angles for each sector
        plt.pie(counts, labels=counts.index, autopct='%.2f')
        plt.title(f"Pie Chart for {col_name}")

        angle_cutoff = 20  # user may edit this.
        if (angles <= angle_cutoff).all():
            print(f"{colored('Notice','red' )}: Profusion of data.")
            print("A Pie chart may not be the most informative visualization for this data.")

        plt.show()

    except IndexError:
        print(f"{colored('Caution!','yellow' )} Pie chart column number {y_col} is out of range.")

# Function to draw a Pair Plot
def draw_pair_plot(df, columns):
    try:
        # Limit the number of columns to 10
        columns = columns[:10]

        diag_kind = input("Enter diag_kind (e.g., 'auto', 'hist', 'kde'): ").strip()
        kind = input("Enter kind (e.g., 'scatter', 'reg', 'hex'): ").strip()

        # Set default values if user presses enter without providing values
        diag_kind = diag_kind if diag_kind else 'auto'
        kind = kind if kind else 'scatter'

        # Convert column names to numbers
        column_numbers = [col_alpha_to_num(col) if not col.isnumeric() else int(col) for col in columns]

        # Check if columns are within bounds
        if any(col < 0 or col >= df.shape[1] for col in column_numbers):
            print(f"{colored('Caution!','yellow' )} One or more columns are out of range.")
            return

        # Check if columns are found in the DataFrame
        invalid_columns = [col for col in column_numbers if col not in range(df.shape[1])]
        if invalid_columns:
            print(f"{colored('Caution!','yellow' )} Column(s) {invalid_columns} not found in the CSV file.")
            return

        # Create pair plot
        sns.pairplot(df.iloc[:, column_numbers], diag_kind=diag_kind, kind=kind)
        plt.show()

    except ValueError as ve:
        print(f"{colored('Caution!','yellow' )} Error in creating pair plot: {ve}")

######################
def draw_correlation_matrix(df, columns_range, variance_threshold=1e-10):
    try:
        start_col, end_col = parse_columns_range(columns_range)
        
        # Check if columns are within bounds
        if not (0 <= start_col < df.shape[1]) or not (0 <= end_col < df.shape[1]):
            print(f"{colored('Caution!','yellow' )} One or more columns are out of range.")
            return

        # Check if columns are found in the DataFrame
        invalid_columns = [col for col in range(start_col, end_col + 1) if col not in range(df.shape[1])]
        if invalid_columns:
            print(f"{colored('Caution!','yellow' )} Column(s) {invalid_columns} not found in the CSV file.")
            return

        # Select the specified columns
        selected_df = df.iloc[:, start_col:end_col + 1]
        
        # Keep only numeric columns
        numeric_df = selected_df.select_dtypes(include='number')
        
        if numeric_df.empty:
            print(f"{colored('Caution!','yellow' )} No numeric columns found in the selected range.")
            return
        
        # Remove columns with variance below the threshold
        low_variance_cols = numeric_df.var() <= variance_threshold
        filtered_df = numeric_df.loc[:, ~low_variance_cols]
        
        if filtered_df.empty:
            print(f"{colored('Caution!','yellow' )} All selected columns have low variance and have been removed.")
            return

        # Calculate the correlation matrix
        correlation_matrix = filtered_df.corr(numeric_only=True)
        
        # Determine whether to annotate cells based on the number of columns
        annot = correlation_matrix.shape[1] <= 45
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', fmt='.2f')

        # Automatically adjust the plot layout to make sure labels are visible
        plt.title("Correlation Matrix (Heatmap)")
        
        plt.tight_layout()  # Makes an initial adjustment
        plt.draw()  # Update the plot
        
        # Get the longest label length on the x-axis
        max_label_length = max([len(str(label)) for label in ax.get_xticklabels()])
        
        # Adjust the plot margins if labels are long
        if max_label_length > 8:
            plt.subplots_adjust(bottom=0.25, left=0.25)
        
        # Tilt x-axis labels if they are longer than 12 characters
        if max_label_length > 12:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.show()

    except ValueError as ve:
        print(f"{colored('Caution!','yellow' )} Error in creating correlation matrix: {ve}")

# Helper function to parse columns range input
def parse_columns_range(columns_range):
    start_col, end_col = map(lambda x: int(x) if x.isnumeric() else col_alpha_to_num(x), columns_range.split(','))
    return start_col, end_col




##############
# Convert alphabetical notation to column number
def col_alpha_to_num(alpha):
    result = 0
    for char in alpha:
        result = result * 26 + ord(char) - ord('A') + 1
    return result - 1

# Main program
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
