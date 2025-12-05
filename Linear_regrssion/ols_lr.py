import pandas as pd
# Define the path to your CSV file
df = pd.read_csv("salary_dataset")
df.head()
df.columns
x=df['YearsExperience']
y=df["Salary"]
import numpy as np
x_mean= np.mean(x)
y_mean=np.mean(y)
n=len(x)
num=0
den=0
for i in range(n):
    num+=(x[i]-x_mean)*(y[i]-y_mean)
    den+=(x[i]-x_mean)**2
slope=num/den
intercept=y_mean-(slope*x_mean)
print(f"slope is {slope}")
print(f"intercept is {intercept}")
print(f"\nThe best-fit line is: Salary = {intercept:.2f} + {slope:.2f} * YearsExperience")
def predict(x_value):
    """Predicts a y value based on the calculated slope and intercept."""
    return intercept + slope * x_value

# Predict the salary for someone with 7 years of experience
years_new = 7
predicted_salary = predict(years_new)
print(f"\nPredicted salary for {years_new} years of experience: ${predicted_salary:.2f}")
# --- 4. Visualization ---
import matplotlib.pyplot as plt
# Create a line of predicted y values
predictions = predict(x)

# Create a scatter plot of the actual data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Data Points')

# Plot the OLS regression line
plt.plot(x, predictions, color='red', linewidth=2, label='OLS Regression Line (from scratch)')

# Add titles and labels for clarity
plt.title('Salary vs. Years of Experience (Manual OLS)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()