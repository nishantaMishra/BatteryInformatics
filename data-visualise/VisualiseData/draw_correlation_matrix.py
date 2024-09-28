import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from utils import parse_columns_range

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

