from termcolor import colored

# Function to check and inform about missing values in specified columns
def check_missing_values(df, x_col, y_cols):
    warnings = []
    if not (0 <= x_col < df.shape[1]):
        warnings.append(f"{colored('Caution!','yellow' )} x-axis column {x_col} is out of range.")
    for y_col in y_cols:
        if not (0 <= y_col < df.shape[1]):
            warnings.append(f"{colored('Caution!','yellow' )} y-axis column {y_col} is out of range.")
    for warning in warnings:
        print(warning)
    return warnings

def col_alpha_to_num(alpha):
    result = 0
    for char in alpha:
        result = result * 26 + ord(char) - ord('A') + 1
    return result - 1

# Helper function to parse columns range input
def parse_columns_range(columns_range):
    start_col, end_col = map(lambda x: int(x) if x.isnumeric() else col_alpha_to_num(x), columns_range.split(','))
    return start_col, end_col

# Convert alphabetical notation to column number
def col_alpha_to_num(alpha):
    result = 0
    for char in alpha:
        result = result * 26 + ord(char) - ord('A') + 1
    return result - 1
