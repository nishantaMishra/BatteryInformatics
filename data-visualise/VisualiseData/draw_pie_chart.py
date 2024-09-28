import matplotlib.pyplot as plt
from termcolor import colored

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
