import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stallTorqueCalc3 import *
from scipy.special import comb

def bezierCurve(controlPoints, numberOfPoints):
    """
    Generates N points approximating a Bezier curve of any degree
    
    Parameters:
        controlPoints (array-like): list of (radius, angle) control points
        numberOfPoints (int): Number of points N to evaluate

    Returns:
        np.ndarray: Array of shape (N, D) containing the sampled curve points

    """

    points = np.asarray(controlPoints)
    n = len(points) - 1 # bezier curve degree

    # generate N t values linearly spaced from 0 to 1
    t = np.linspace(0, 1, numberOfPoints)[:, np.newaxis]

    # calculate bernstein polynomial weights
    i = np.arange(n + 1)

    # binomial coefficients
    weights = comb(n, i) * ((1 - t) ** (n - i)) * (t ** i)

    # matrix multiplication to combine weights with control points
    return np.dot(weights, points)

bucketWallWidth = 0.01 # m
outerRadius = 0.5 # m
bezierDegree = 3
numberOfBucketPathPoints = 5


# initialise radii and angle arrays
radii = np.linspace(0, outerRadius, bezierDegree)
angles = np.zeros(bezierDegree)

# convert radii and angles to bucket path array
def getControlPoints(initialRadii, initialAngles):
    controlPoints = []
    controlPoints.append([initialRadii[0], 0])
    for i in range(bezierDegree - 2):
        controlPoints.append([initialRadii[i + 1], initialAngles[i + 1]])
    controlPoints.append([outerRadius, initialAngles[-1]])
    return controlPoints

def testStallTorque(radii, angles, plotResult=False):


    controlPoints = getControlPoints(radii, angles)

    curvePointsPolar = bezierCurve(controlPoints, numberOfBucketPathPoints)

    try:
        stallTorque = getAutoStallTorquePerMetre(curvePointsPolar, bucketWallWidth, False, plotResult)
    except:
        stallTorque = 0.0

    return stallTorque


numberOfOptimisationSteps = 100
dr = 0.01 # m
dth = 0.01 # rad
stallTorqueDiffArr = []
previousStallTorque = testStallTorque(radii, angles)
print(previousStallTorque)
for i in range(numberOfOptimisationSteps):
    # vary radii arrays
    for j in range(bezierDegree):
        radii[j] += dr
        currentStallTorque = testStallTorque(radii, angles)
        if currentStallTorque < previousStallTorque:
            radii[j] -= 2*dr

        radii[j] = min(max(radii[j], 0), outerRadius)

    # vary angle arrays
    for j in range(bezierDegree):
        angles[j] += dth
        currentStallTorque = testStallTorque(radii, angles)
        if currentStallTorque < previousStallTorque:
            angles[j] -= 2*dth

    print(f"===== ROUND {i} =====")
    print("radii:", radii, "m")
    print("angles:", angles, "rad")
    print("stall torque per metre:", round(currentStallTorque, 1), "Nm/m")

    testStallTorque(radii, angles, True)
    

    

