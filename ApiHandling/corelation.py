import csv
from decimal import Decimal
import pandas as pd
import numpy as np
import json
from datetime import datetime
from fancyimpute import IterativeImputer
from sklearn.impute import KNNImputer
# Load JSON file
with open('response.json', 'r') as file:
    data = json.load(file)

dependant_variable = 'EntityCommonStockSharesOutstanding'

# Helper function to convert string to datetime object
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None

def get_share_of_dependant_var(date):
    try:
        # Attempt to parse as YYYY-MM-DD
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        # If parsing fails, return None or raise an error
        print(f"Error: The date '{date}' is not in the correct format.")
        return None

    # Format the date as DD.MM.YYYY for matching
    formatted_date = parsed_date.strftime("%d.%m.%Y")
    #print("----FORMATTED DATA------>", formatted_date)
    
    with open('aaple_datas.csv', 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)  # Use DictReader to get a dictionary for each row
        closest_date = None
        closest_value = None
        min_diff = float('inf')
        
        for row in reader:
            row_date = datetime.strptime(row['Date'], "%d.%m.%Y")
            if row['Date'] == formatted_date:
                result = row['Now']  # Return the "Now" value for the specified date
                if result is not None:
                    return result
            else:
                # Calculate the difference in days
                diff = abs((row_date - parsed_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_date = row_date
                    closest_value = row['Now']
        
        if closest_value is not None:
            print(f"Closest date found: {closest_date.strftime('%d.%m.%Y')} with value: {closest_value}")
            return closest_value
    
    return None  # Return None if the date is not found


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
                            if end_date not in parameters:
                                parameters[end_date] = {}
                            if sub_category == dependant_variable and dependant_variable=='EntityCommonStockSharesOutstanding':
                                share = get_share_of_dependant_var(str(end_date))
                                share_decimal = Decimal(share.replace(',', '.'))  # ',' yerine '.' kullan
                                value_decimal = Decimal(value)  # Direk stringi Decimal'e çevir
                                value = share_decimal * value_decimal #reg3.txt
                                #print("----Share---->", share_decimal)
                                #print("----Value---->", value_decimal)
                                #print("----Company Value---->", value)
                            parameters[end_date][sub_category] = value

# Convert to DataFrame
df = pd.DataFrame.from_dict(parameters, orient='index')
# Check if dependant_variable exists in the columns
if dependant_variable not in df.columns:
    raise ValueError("dependant_variable not found in the original data.")

# Sort the DataFrame by date
df = df.sort_index()
with open('DF.txt', 'w') as result_file:
    for index in df.index:
        result_file.write(f"{index}\n")
# Convert all columns to numeric, coercing errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate the percentage of non-NaN values for each column
non_nan_percentage = df.count() / len(df) * 100

# Set the threshold for keeping columns to 75%
threshold = 5
filtered_df = df.loc[:, non_nan_percentage >= threshold]

# Handle remaining missing values by filling with the mean of each column 
# commandler degistirerek sonuclar test edildi 
#filtered_df = filtered_df.fillna(filtered_df.mean())
#filtered_df = filtered_df.interpolate(method='linear')
#imputer = IterativeImputer() #reg.txt
imputer = KNNImputer(n_neighbors=5) #reg2.txt
filtered_df = pd.DataFrame(imputer.fit_transform(filtered_df), columns=filtered_df.columns)
# Check if dependant_variable is still in the filtered DataFrame
if dependant_variable in filtered_df.columns:
    # Calculate correlation matrix with higher precision
    correlation_matrix = filtered_df.corr(method='pearson', min_periods=1)

    # Extract correlation with the dependent variable
    dependent_variable_correlation = correlation_matrix[dependant_variable].sort_values(ascending=False)

    # Write results to a file
    filename = f"correlation_results_{dependant_variable}_DENEME_reg3.txt"
    with open(filename, 'w') as result_file:
        result_file.write(f"Correlation Results:\n{dependent_variable_correlation.to_string(float_format='{:.6f}'.format)}\n")
        result_file.write(f"\nNumber of Variables: {len(filtered_df.columns)}\n")
        result_file.write(f"\nShape of filtered DataFrame: {filtered_df.shape}\n")
        result_file.write(f"\nColumns removed due to insufficient data: {list(set(df.columns) - set(filtered_df.columns))}\n")
        result_file.write(f"\nDate range of analysis for {dependant_variable}: {filtered_df.index.min()} to {filtered_df.index.max()}\n")
    
    print("Results written to {filename}")
else:
    print(f"Warning: {dependant_variable} was removed during filtering (threshold: {threshold}%).")
    print(f"Non-NaN percentage for {dependant_variable}: {non_nan_percentage[dependant_variable]:.2f}%")

# Print additional information for debugging
print(f"\nShape of original DataFrame: {df.shape}")
print(f"Shape of filtered DataFrame: {filtered_df.shape}")
print(f"Date range of original data: {df.index.min()} to {df.index.max()}")

# Print non-NaN percentages for all columns
print("\nNon-NaN percentages for all columns:")
#for col, percentage in non_nan_percentage.items():
#    print(f"{col}: {percentage:.2f}%")

# Analyze dependent variable data
dep_data = df[dependant_variable].dropna()
print(f"\n{dependant_variable} data points: {len(dep_data)}")
print(f"{dependant_variable} date range: {dep_data.index.min()} to {dep_data.index.max()}")
print(f"{dependant_variable} mean value: {dep_data.mean():.2f}")
print(f"{dependant_variable} median value: {dep_data.median():.2f}")



""" code below was for pearson manual calculation - handled as corr() function above -
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Load JSON file
with open('response.json', 'r') as file:
    data = json.load(file)

dependant_variable = 'Assets'

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
                            if end_date not in parameters:
                                parameters[end_date] = {}
                            parameters[end_date][sub_category] = value

# Convert to DataFrame
df = pd.DataFrame.from_dict(parameters, orient='index')

# Check if dependant_variable exists in the columns
if dependant_variable not in df.columns:
    raise ValueError(f"{dependant_variable} not found in the original data.")

# Sort the DataFrame by date
df = df.sort_index()

# Convert all columns to numeric, coercing errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate the percentage of non-NaN values for each column
non_nan_percentage = df.count() / len(df) * 100

# Lower the threshold for keeping columns to 25%
threshold = 25
filtered_df = df.loc[:, non_nan_percentage >= threshold]

# Handle remaining missing values by filling with the mean of each column
filtered_df = filtered_df.fillna(filtered_df.mean())

# Custom function to calculate Pearson correlation between two variables
def custom_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    
    if denominator == 0:
        return np.nan  # Avoid division by zero
    else:
        return numerator / denominator

# Check if dependant_variable is still in the filtered DataFrame
if dependant_variable in filtered_df.columns:
    dependent_variable_data = filtered_df[dependant_variable]
    
    # Calculate correlation manually for each variable in the filtered DataFrame
    correlations = {}
    for column in filtered_df.columns:
        if column != dependant_variable:
            correlations[column] = custom_correlation(filtered_df[column], dependent_variable_data)
    
    # Sort correlations by value
    sorted_correlations = dict(sorted(correlations.items(), key=lambda item: item[1], reverse=True))
    
    # Write results to a file
    filename = f"manual_correlation_results_{dependant_variable}_DENEME.txt"
    with open(filename, 'w') as result_file:
        result_file.write(f"Correlation Results (Manual Calculation):\n")
        result_file.write(f"{dependant_variable}: 1.000000\n")
        for col, corr in sorted_correlations.items():
            result_file.write(f"{col}: {corr:.6f}\n")
        result_file.write(f"\nNumber of Variables: {len(filtered_df.columns)}\n")
        result_file.write(f"\nShape of filtered DataFrame: {filtered_df.shape}\n")
        result_file.write(f"\nColumns removed due to insufficient data: {list(set(df.columns) - set(filtered_df.columns))}\n")
        result_file.write(f"\nDate range of analysis for {dependant_variable}: {filtered_df.index.min()} to {filtered_df.index.max()}\n")
    
    print(f"Results written to {filename}")
else:
    print(f"Warning: {dependant_variable} was removed during filtering (threshold: {threshold}%).")

# Print additional debugging information
print(f"\nShape of original DataFrame: {df.shape}")
print(f"Shape of filtered DataFrame: {filtered_df.shape}")
print(f"Date range of original data: {df.index.min()} to {df.index.max()}")

# Analyze dependant_variable data
dep_var_data = df[dependant_variable].dropna()
print(f"{dependant_variable} data points: {len(dep_var_data)}")
print(f"{dependant_variable} date range: {dep_var_data.index.min()} to {dep_var_data.index.max()}")
print(f"{dependant_variable} mean value: {dep_var_data.mean():.2f}")
print(f"{dependant_variable} median value: {dep_var_data.median():.2f}")
"""