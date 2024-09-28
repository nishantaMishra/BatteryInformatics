import pandas as pd
import os
from termcolor import colored
import readline
from utils import check_missing_values, col_alpha_to_num, parse_columns_range, col_alpha_to_num
from draw_line_graph import draw_line_graph
from draw_scatter_plot import draw_scatter_plot
from draw_bar_graph import draw_bar_graph
from draw_histogram import draw_histogram
from draw_pie_chart import draw_pie_chart
from draw_pair_plot import draw_pair_plot
from draw_correlation_matrix import draw_correlation_matrix

def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        print("File not found.")
        return None

# Function to enable tab-completion for file paths (tested only in linux)
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


# Function to display the menu
def display_menu():
    options = [
        "1: Line graph", 
        "2: Scatter Plot", 
        "3: Bar graph", 
        "4: Histogram", 
        "5: Pie chart", 
        "6: Pair plot", 
        "7: Correlation Matrix (Heatmap)", 
        "0: Quit"
    ]
    print("")
    print("================>>> Utilities <<<===============")
    # Print in two columns
    for i in range(0, len(options), 2): # Display the menu in two columns
        left_column = options[i]
        right_column = options[i + 1] if i + 1 < len(options) else ""  # Ensure no IndexError
        print(f"{left_column.ljust(34)} {right_column}")

    print("------------>>")



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

            """print("Choose a plot type:")
            print("1: Line graph")
            print("2: Scatter Plot")
            print("3: Bar graph")
            print("4: Histogram")
            print("5: Pie chart")
            print("6: Pair plot")
            print("7: Correlation Matrix (Heatmap)")
            print("0: Quit")
            print("------------>>")"""

            # Display the menu
            display_menu()

            choice = input("Enter your choice: ")

            if choice == '1':
                print("-------------------->> Line Plot <<--------------------")
                x_col_input = input("Enter the x-axis column (number or alphabetical notation): ")
                x_col = int(x_col_input) if x_col_input.isnumeric() else col_alpha_to_num(x_col_input)
                y_cols_input = input("Enter the y-axis column(s) (comma-separated): ")
                y_cols = [int(y) if y.isnumeric() else col_alpha_to_num(y) for y in y_cols_input.split(',')]
                draw_line_graph(df, x_col, y_cols)
            elif choice == '2':
                print("-------------------->> Scatter Plot <<--------------------")
                x_col_input = input("Enter the x-axis column (number or alphabetical notation): ")
                x_col = int(x_col_input) if x_col_input.isnumeric() else col_alpha_to_num(x_col_input)
                y_cols_input = input("Enter the y-axis column(s) (comma-separated): ")
                y_cols = [int(y) if y.isnumeric() else col_alpha_to_num(y) for y in y_cols_input.split(',')]
                draw_scatter_plot(df, x_col, y_cols)
            elif choice == '3':
                print("-------------------->> Bar Graph <<--------------------")
                x_col_input = input("Enter the x-axis column (number or alphabetical notation): ")
                x_col = int(x_col_input) if x_col_input.isnumeric() else col_alpha_to_num(x_col_input)
                y_cols_input = input("Enter the y-axis column(s) (comma-separated): ")
                y_cols = [int(y) if y.isnumeric() else col_alpha_to_num(y) for y in y_cols_input.split(',')]
                draw_bar_graph(df, x_col, y_cols)
            elif choice == '4':
                print("-------------------->> Histogram <<--------------------")
                y_cols_input = input("Enter the column(s) for the histogram (comma-separated): ")
                y_cols = [int(y) if y.isnumeric() else col_alpha_to_num(y) for y in y_cols_input.split(',')]
                draw_histogram(df, y_cols)
            elif choice == '5':
                print("-------------------->> Pie Chart <<--------------------")
                y_col_input = input("Enter the column for the Pie chart (number or alphabetical notation): ")
                y_col = [int(y_col_input) if y_col_input.isnumeric() else col_alpha_to_num(y_col_input)]
                draw_pie_chart(df, y_col)
            elif choice == '6':
                print("-------------------->> Pair Plot <<--------------------")
                columns_input = input("Enter columns for the pair plot (comma-separated): ")
                columns = [col.strip() for col in columns_input.split(',')]
                draw_pair_plot(df, columns)
            elif choice == '7':
                print("-------------------->> Correlation Matrix <<--------------------")
                columns_range_input = input("Enter the range of columns for the correlation matrix (e.g., 'A, C' or '1, 3'): ")
                draw_correlation_matrix(df, columns_range_input)
            elif choice == '0':    
                break
            else:
                print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
