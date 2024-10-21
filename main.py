import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('salary.csv')
x = df['YearsExperience'].values
y = df['Salary'].values

# Calculate means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate slope (b1) and intercept (b0)
numerator = 0
denominator = 0

for index, row in df.iterrows():
    xi = row['YearsExperience']
    yi = row['Salary']
    numerator += (xi - x_mean) * (yi - y_mean)
    denominator += (xi - x_mean) ** 2

b1 = numerator / denominator  # Slope
b0 = y_mean - (b1 * x_mean)  # Intercept

# Generate y values for the regression line
y_pred = b1 * x + b0

# Create the plot
plt.scatter(x, y, color='blue', label='Data Points')  # Scatter plot for actual data
plt.plot(x, y_pred, color='red', label=f'y = {b1:.2f}x + {b0:.2f}')  # Regression line

# Customize the plot
plt.title('Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.axhline(0, color='black', linewidth=0.5, ls='--')  # x-axis
plt.axvline(0, color='black', linewidth=0.5, ls='--')  # y-axis
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
