import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from termcolor import colored
from utils import check_missing_values

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

            plt.plot(x_curve, y_curve, label=f'Curve Fit (Degree {degree})', linestyle='--')

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