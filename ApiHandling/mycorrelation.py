import pandas as pd
import json
from datetime import datetime

# Load JSON file
with open('response.json', 'r') as file:
    data = json.load(file)

dependant_variable = 'Assets'  # Bağımlı değişken (satır indeksi olarak)

# Helper function to convert string to datetime object
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None

# Extract data including dates
parameters = {}
for category, values in data['facts'].items():
    for sub_category, sub_values in values.items():
        if 'units' in sub_values:
            for unit, unit_values in sub_values['units'].items():
                if isinstance(unit_values, list):
                    for entry in unit_values:
                        end_date = parse_date(entry.get('end'))
                        value = entry.get('val')
                        if end_date:
                            if sub_category not in parameters:
                                parameters[sub_category] = {}
                            parameters[sub_category][end_date] = value

# Convert to DataFrame
df = pd.DataFrame.from_dict(parameters, orient='index')

# Transpose the DataFrame to swap rows and columns

# Print DataFrame after transposing
print("DataFrame after transposing:")
print(df)

# Clean index by stripping whitespace
df.index = df.index.str.strip()
with open('DF.txt', 'w') as result_file:
    for index in df.index:
        result_file.write(f"{index}\n")

# Sort the DataFrame by date (optional)
df = df.sort_index(axis=1)  # Sort columns (dates)

# Convert all columns to numeric, coercing errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate the percentage of non-NaN values for each row
non_nan_percentage = df.count(axis=1) / len(df.columns) * 100

# Set the threshold for keeping rows to 75%
threshold = 40
filtered_df = df.loc[non_nan_percentage >= threshold]

# Handle remaining missing values by filling with the mean of each row
filtered_df = filtered_df.fillna(filtered_df.mean(axis=1), axis=0)

# Check if dependant_variable is still in the filtered DataFrame
if dependant_variable in filtered_df.index:
    # Calculate correlation matrix with higher precision
    print(filtered_df.index)
    correlation_matrix = filtered_df.corr(method='pearson', min_periods=1)
    print(correlation_matrix.index)

    # Extract correlation with the dependent variable
    dependent_variable_correlation = correlation_matrix[dependant_variable].sort_values(ascending=False)

    # Write results to a file
    filename = f"correlation_results_{dependant_variable}_DENEME_reg2_col.txt"
    with open(filename, 'w') as result_file:
        result_file.write(f"Correlation Results:\n{dependent_variable_correlation.to_string(float_format='{:.6f}'.format)}\n")
        result_file.write(f"\nNumber of Variables: {len(filtered_df.index)}\n")
        result_file.write(f"\nShape of filtered DataFrame: {filtered_df.shape}\n")
        result_file.write(f"\nRows removed due to insufficient data: {list(set(df.index) - set(filtered_df.index))}\n")
        result_file.write(f"\nDate range of analysis for {dependant_variable}: {filtered_df.columns.min()} to {filtered_df.columns.max()}\n")

    print(f"Results written to {filename}")
else:
    print(f"Warning: {dependant_variable} was removed during filtering (threshold: {threshold}%).")
    print(f"Non-NaN percentage for {dependant_variable}: {non_nan_percentage[dependant_variable]:.2f}%")

# Print additional information for debugging
print(f"\nShape of original DataFrame: {df.shape}")
print(f"Shape of filtered DataFrame: {filtered_df.shape}")
print(f"Date range of original data: {df.columns.min()} to {df.columns.max()}")

# Print non-NaN percentages for all rows
print("\nNon-NaN percentages for all rows:")
for index, percentage in non_nan_percentage.items():
    print(f"{index}: {percentage:.2f}%")

# Print the filtered DataFrame for inspection
print("\nFiltered DataFrame:")
print(filtered_df)

# Analyze dependent variable data
dep_data = df.loc[dependant_variable].dropna()
print(f"\n{dependant_variable} data points: {len(dep_data)}")
print(f"{dependant_variable} date range: {dep_data.index.min()} to {dep_data.index.max()}")
print(f"{dependant_variable} mean value: {dep_data.mean():.2f}")
print(f"{dependant_variable} median value: {dep_data.median():.2f}")
