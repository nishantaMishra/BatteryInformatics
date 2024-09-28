import matplotlib.pyplot as plt
from termcolor import colored

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