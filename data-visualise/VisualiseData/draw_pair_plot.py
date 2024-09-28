import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from utils import col_alpha_to_num

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