import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stallTorqueCalc import getStallTorquePerMetre, getBucketWallAreaRatio, getIdealNumberOfBuckets

bucketWallWidth = 0.01 # m

data = {
    "x": [],
    "y": [],
}

# DO TEST
innerRadiusStart = 0.15
innerRadiusEnd = 0.35
numberOfTests = 10
innerRadiusArr = np.linspace(innerRadiusStart, innerRadiusEnd, numberOfTests) # [m]
outerRadius = 0.5 # [m]
bucketAngle = 0.6 # [rad]
flatLength = 0.1
for innerRadius in innerRadiusArr:
    wallPathPolar = ((innerRadius, 0), (innerRadius+flatLength, 0), (outerRadius, bucketAngle))
    numberOfBuckets = getIdealNumberOfBuckets(wallPathPolar, bucketWallWidth)

    try:
        stallTorquePerMetre = getStallTorquePerMetre(wallPathPolar, numberOfBuckets, True)
    except AttributeError:
        continue

    data["x"].append(innerRadius)
    data["y"].append(stallTorquePerMetre)

# CALCULATE POLYFIT
degree = 2
coefficients = np.polyfit(data["x"], data["y"], degree)
polyFunc = np.poly1d(coefficients)
xSmooth = np.linspace(innerRadiusStart, innerRadiusEnd, numberOfTests)
ySmooth = polyFunc(xSmooth)

# SAVE DATA

# Convert dictionary to a pandas DataFrame
df = pd.DataFrame(data)

print(df)

# Save the dataset to a .csv file
csv_filename = "IRTest.csv"
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
plt.title("Stall torque generated as the wheel inner radius varies")
plt.xlabel("Inner radius [m]")
plt.ylabel("Stall torque per metre [Nm/m]")

# Add grid lines for readability
plt.grid(True, alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()