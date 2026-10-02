import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
import math
from typing import Any



def polarToCartesian(r, thetaRad) -> tuple:
    x = r * np.cos(thetaRad)
    y = r * np.sin(thetaRad)
    return (x, y)

def cartesianToPolar(x, y) -> tuple:
    r = np.hypot(x, y)
    thetaRad = np.arctan2(y, x)

    return (r, thetaRad)

def pathLength(points):
    pts = np.asarray(points)
    return np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))

def rotate(v, thetaRad):
    v = np.asarray(v)
    R = np.array([
        [np.cos(thetaRad), -np.sin(thetaRad)],
        [np.sin(thetaRad),  np.cos(thetaRad)]
    ])
    return R @ v

def polygonArea(points):
    points = np.asarray(points)

    if len(points) <= 2:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]
    
    # roll shifts arrays by 1 (wrap-around)
    area = 0.5 * np.abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )
    
    return area

def polygonCentroid(points):
    points = np.asarray(points, dtype=float)

    if len(points) < 3:
        raise ValueError("Polygon requires at least 3 points")

    x = points[:, 0]
    y = points[:, 1]

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross)

    if abs(signed_area) < 1e-12:
        raise ValueError("Polygon has zero area")

    cx = np.sum((x + x_next) * cross) / (6 * signed_area)
    cy = np.sum((y + y_next) * cross) / (6 * signed_area)

    return np.array([cx, cy])


def getBucketWallAreaRatio(wallPathPolar, bucketWallWidth, numberOfBuckets: int):
    # wheel geometry
    outerRadius = wallPathPolar[-1][0]
    innerRadius = wallPathPolar[0][0]

    wallPathCart = tuple(polarToCartesian(r, thetaRad) for r, thetaRad in wallPathPolar)
    annulusArea = np.pi * (outerRadius**2 - innerRadius**2)

    

    bucketWallArea = (
        pathLength(wallPathCart)
        * bucketWallWidth
        * numberOfBuckets
    )

    bucketWallAreaRatio = bucketWallArea / annulusArea

    if bucketWallAreaRatio >= 0.1:
        print(f"Warning! Bucket walls take up {round(bucketWallAreaRatio,3)*100}% of the annulus area. Rule of thumb is below 10%")

    return bucketWallAreaRatio

def getStallTorquePerMetre(wallPathPolar, numberOfBuckets: int, plotResult=False):

    # check thta first angle is 0
    if wallPathPolar[0][1] != 0:
        raise ValueError(f"First point in wallPathPolar should have angle = 0. It is now {wallPathPolar[0][1]} rad")

    numberOfWallPoints = len(wallPathPolar)

    # wheel geometry
    outerRadius = wallPathPolar[-1][0]
    innerRadius = wallPathPolar[0][0]

    


    # calculate bucket points
    bucketPointsCart = np.zeros((numberOfBuckets, numberOfWallPoints, 2)) # For each bucket wall, stores xy points in m
    bucketPointsPolar = np.zeros((numberOfBuckets, numberOfWallPoints, 2)) # For each bucket wall, stores xy points in m

    for i in range(numberOfBuckets):
        wheelThetaRad = i * np.pi * 2 / numberOfBuckets

        for j , (r, bucketThetaRad) in enumerate(wallPathPolar):

            # polar
            bucketPointsPolar[i][j][0] = r
            bucketPointsPolar[i][j][1] = bucketThetaRad + wheelThetaRad

            # cartesian
            bucketPointsCart[i][j] = polarToCartesian(r, bucketThetaRad + wheelThetaRad)


    def getBucketPointFormat(x, y, isLeft: bool, isOutside: bool):
        """convert points to dicts to keep left/right and isOutside info"""
        return {"xy": (x,y), "x": x, "y": y, "left?": isLeft, "outside?": isOutside}

    # fill buckets
    waterPolyList: list[Any] = [None] * numberOfBuckets
    waterPolyCentroidList: list[Any] = [None] * numberOfBuckets
    bucketAreas = np.zeros((numberOfBuckets))
    stallTorqueAccumulator = 0 #Nm/m
    stallTorqueContributions = np.zeros((numberOfBuckets)) #Nm/m
    for i in range(numberOfBuckets):


        # check that filling this bucket would not create an anti-clockwise torque
        wheelThetaRad = i * np.pi * 2 / numberOfBuckets

        if np.pi/2 < wheelThetaRad < 3*np.pi/2:
            continue

        # get the right and left bucket paths 
        bucketWallPathRightCart = bucketPointsCart[i]
        if i == numberOfBuckets - 1:
            bucketWallPathLeftCart = bucketPointsCart[0]
        else:
            bucketWallPathLeftCart = bucketPointsCart[i+1]

        # convert points to dicts to keep left/right and isOutside info
        bucketPoly = []
        for j, (x, y) in enumerate(bucketWallPathRightCart):

            if j == numberOfWallPoints - 1:

                bucketPoly.append(getBucketPointFormat(x, y, False, True))

            else:

                bucketPoly.append(getBucketPointFormat(x, y, False, False))

        for j, (x, y) in enumerate(bucketWallPathLeftCart):

            if j == numberOfWallPoints - 1:
            
                bucketPoly.append(getBucketPointFormat(x, y, True, True))

            else:

                bucketPoly.append(getBucketPointFormat(x, y, True, False))

        # Get a sorted list of points by elevation
        bucketPolySorted = sorted(bucketPoly, key=lambda p: p["y"])

        # Find the point at which the water spills out
        spillPointIndex = None

        for j, item in enumerate(bucketPolySorted):

            if item["outside?"]:
                spillPointIndex = j
                spillPoint = item
                isSpillPointOnLeft = item["left?"]
                break

        if spillPointIndex is None:
            raise ValueError(f"No spill point found for bucket {i}!")

        if spillPointIndex == 2*numberOfWallPoints - 1:
            print("Warning! spill point is at the very top...")


        # get the 2 points above/below spill point, on the other side of the spill point
        pointsAboveSpill = bucketPolySorted[spillPointIndex+1:]
        pointsBelowSpill = bucketPolySorted[:spillPointIndex][::-1]

        # bucket is empty when the spill point is at the lowest point
        if len(pointsBelowSpill) == 0:
            continue

        # Find the line segment on the other side of the spill point
        # This is defined between the offside points above/below
        offsidePointAbove = None
        for item in pointsAboveSpill:
            if item["left?"] != isSpillPointOnLeft:
                offsidePointAbove = item
                break

        if offsidePointAbove is None:
            print("Warning! no offside point found above spill point...")
            raise Exception

        offsidePointBelow = None
        for item in pointsBelowSpill:
            if item["left?"] != isSpillPointOnLeft:
                offsidePointBelow = item
                break
    
        # special case where water line runs up to inner circle
        if offsidePointBelow is None:
            x, y = bucketWallPathRightCart[0]
            offsidePointBelow = getBucketPointFormat(x, y, False, False)


        # start populating the waterPoly    
        # going anti-clockwise from the inner right point
        waterPoly = []

        # add the points for the bucket's right side
        for point in bucketWallPathRightCart:
            if point[1] >= spillPoint["y"]:
                continue

            waterPoly.append(point)

        if not isSpillPointOnLeft:
            waterPoly.append(np.asarray(spillPoint["xy"]))


        # add intersection of waterline
        if abs(offsidePointAbove["y"] - offsidePointBelow["y"]) < 1e-8:
            continue
        t = (spillPoint["y"] - offsidePointBelow["y"]) / (offsidePointAbove["y"] - offsidePointBelow["y"])
        intersectionX = offsidePointBelow["x"] + t * (offsidePointAbove["x"] - offsidePointBelow["x"])
        intersectionY = spillPoint["y"]
        waterPoly.append(np.asarray((intersectionX, intersectionY)))

        if isSpillPointOnLeft:
            waterPoly.append(np.asarray(spillPoint["xy"]))


        # add the points for the bucket's left side
        for point in bucketWallPathLeftCart[::-1]:
            if point[1] >= spillPoint["y"]:
                continue

            waterPoly.append(point)

        # centroid (centre of mass) of water in bucket
        waterPolyCentroid = polygonCentroid(waterPoly)

        # check that water doesn't create anti-clockwise torque
        if waterPolyCentroid[0] < 0:
            continue

        waterPolyCentroidPolar = cartesianToPolar(waterPolyCentroid[0], waterPolyCentroid[1])

        # check that centroid is in the annulus 
        if waterPolyCentroidPolar[0] > outerRadius or waterPolyCentroidPolar[0] < innerRadius:
            continue

        # save data
        waterPolyList[i] = waterPoly.copy()
        waterPolyCentroidList[i] = waterPolyCentroid

        waterArea = polygonArea(waterPoly)
        bucketAreas[i] = waterArea

        # add stall torque contribution
        stallTorqueContribution = waterArea * 1000 * 9.81 * waterPolyCentroid[0]
        stallTorqueContributions[i] = stallTorqueContribution
        stallTorqueAccumulator += stallTorqueContribution

    if plotResult:
        fig, ax = plt.subplots(figsize=(8,8))

        # Plot bucket walls
        for i in range(numberOfBuckets):
            ax.plot(
                bucketPointsCart[i, :, 0],
                bucketPointsCart[i, :, 1],
            )

        # Draw inner and outer circles
        ax.add_patch(Circle((0, 0), innerRadius, fill=False))
        ax.add_patch(Circle((0, 0), outerRadius, fill=False))

        # draw water polygons
        for poly in waterPolyList:
            if poly is None:
                continue
            polygon = Polygon(poly, closed=True, alpha=0.3)
            ax.add_patch(polygon)

        # add labels
        for i in range(numberOfBuckets):
            x, y = bucketPointsCart[i][0]
            y += 0.01
            ax.text(x, y, f"{i}")

        # add centroids
        for i, point in enumerate(waterPolyCentroidList):
            if point is None:
                continue
            x, y = point
            ax.plot(x, y, marker="+", markersize=14, markeredgewidth=1, color=(0,0,0.5, 1))
            ax.text(x+0.01, y+0.01, f"Bucket {i}", fontsize=12)
            ax.text(x+0.01, y, f"{round(bucketAreas[i] * 1000 * 9.81, 1)} N/m", fontsize=8)
            ax.text(x+0.01, y-0.01, f"{round(stallTorqueContributions[i], 1)} Nm/m", fontsize=8)


        # centre mark
        ax.plot(0, 0, marker="+", markersize=32, markeredgewidth=2, color=(0,0,0, .5))


        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid()

        plt.show()


    return stallTorqueAccumulator

"""
Define bucket wall geometry using polar coordinates

Points are written in polar coordinates (r, theta)
    - Radius r: distance from wheel centre [m]
    - theta: counter-clockwise angle from +x axis [rad]

Example of bucket with radial side, then angled:
(0.4, 0)
(0.45, 0)
(0.5, 1)

    - The radius of the inner circle is given by the r value of the first point
    - The angle of the first point should be 0 
"""

wallPathPolar1 = ((0.30, 0),
                 (0.35, 0.05),
                 (0.40, 0.10),
                 (0.45, 0.15),
                 (0.50, 0.20),
                 (0.55, 0.25),)

wallPathPolar2 = ((0.10, 0),
                  (0.20, 0),
                  (0.50, 0.40))

torquePerMetre = getStallTorquePerMetre(wallPathPolar1, 32, True)
print(torquePerMetre)
    
wallAreaRatio = getBucketWallAreaRatio(wallPathPolar1, 0.01, 32)
print(wallAreaRatio)