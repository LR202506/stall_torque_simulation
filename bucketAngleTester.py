import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stallTorqueCalc import *

bucketWallWidth = 0.01 # m

data = {
    "x": [],
    "y": [],
}

# init variables
innerRadius = 0.3
outerRadius = 0.5 # [m]
numberOfTests = 30
bucketAngle_1 = 0 # [rad]
bucketAngle_2 = 1 # [rad]
bucketAngleArr = np.linspace(bucketAngle_1, bucketAngle_2, numberOfTests) # [m]

# Do tests
for bucketAngleVar in bucketAngleArr:
    wallPathPolar = ((innerRadius, 0), (outerRadius, bucketAngleVar))
    numberOfBuckets = getIdealNumberOfBuckets(wallPathPolar, bucketWallWidth)

    try:
        stallTorquePerMetre = getAutoStallTorquePerMetre(wallPathPolar, bucketWallWidth, False, False, 5)
    except AttributeError:
        continue

    data["x"].append(bucketAngleVar)
    data["y"].append(stallTorquePerMetre)

# CALCULATE POLYFIT
degree = 2
coefficients = np.polyfit(data["x"], data["y"], degree)
polyFunc = np.poly1d(coefficients)
xSmooth = np.linspace(bucketAngle_1, bucketAngle_2, numberOfTests)
ySmooth = polyFunc(xSmooth)

# SAVE DATA

# Convert dictionary to a pandas DataFrame
df = pd.DataFrame(data)

print(df)

# Save the dataset to a .csv file
csv_filename = "angleTest.csv"
df.to_csv(csv_filename, index=False)
print(f"Data successfully saved to '{csv_filename}'")


# PLOT DATA

# Create the scatter plot using Matplotlib
plt.figure(figsize=(8, 5))
plt.scatter(
    df["x"],
    df["y"],
    color="blue",
)
plt.plot(xSmooth, ySmooth, color="red", label=f"Polyfit, degree {degree}")



# Add titles and labels
plt.title("Stall torque generated as the bucket angle varies")
plt.xlabel("Bucket angle [rad]")
plt.ylabel("Stall torque per metre [Nm/m]")

# Add grid lines for readability
plt.grid(True, alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()