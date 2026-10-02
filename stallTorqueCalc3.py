import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon, box
from typing import Any

def get_bucket_water_polygon(bucket_right_cart, bucket_left_cart):
    """
    Constructs the solid bucket geometry and clips it to the maximum fill level (spill point).
    
    bucket_right_cart: Nx2 numpy array of (x, y) coordinates for the right wall
    bucket_left_cart: Nx2 numpy array of (x, y) coordinates for the left wall
    """
    # 1. Build a closed polygon loop for the empty bucket shell
    # Right wall (inner to outer) -> Left wall reversed (outer to inner)
    bucket_vertices = np.vstack([
        bucket_right_cart,
        bucket_left_cart[::-1]
    ])
    
    bucket_poly = ShapelyPolygon(bucket_vertices)
    if not bucket_poly.is_valid:
        bucket_poly = bucket_poly.buffer(0) # Fix minor self-intersections if any

    # 2. Identify the spill height (lowest y-value among the outer rim points)
    spill_y = min(bucket_right_cart[-1][1], bucket_left_cart[-1][1])

    # 3. Create a clipping box from below the bucket up to spill_y
    # Use bounds of the bucket to ensure the box covers the full width
    minx, miny, maxx, maxy = bucket_poly.bounds
    
    # If the spill point is below the bottom of the bucket, it holds no water
    if spill_y <= miny:
        return None, 0.0, None

    water_clip_box = box(minx - 1.0, miny - 1.0, maxx + 1.0, spill_y)

    # 4. Perform boolean intersection to get the contained water body
    water_poly = bucket_poly.intersection(water_clip_box)

    if water_poly.is_empty or water_poly.area < 1e-8:
        return None, 0.0, None

    # Handle GeometryCollection and MultiPolygon types from Shapely
    if hasattr(water_poly, 'geoms'):
        polygons = [g for g in water_poly.geoms if g.geom_type == 'Polygon'] # type: ignore
        if not polygons:
            return None, 0.0, None
        water_poly = max(polygons, key=lambda p: p.area)

    # 5. Extract centroid and area directly
    centroid_x, centroid_y = water_poly.centroid.x, water_poly.centroid.y
    water_area = water_poly.area

    return water_poly, water_area, (centroid_x, centroid_y)

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
        * numberOfBuckets)

    bucketWallAreaRatio = bucketWallArea / annulusArea

    return bucketWallAreaRatio

def getIdealNumberOfBuckets(wallPathPolar, bucketWallWidth, idealAreaRatio=0.1):
    # wheel geometry
    outerRadius = wallPathPolar[-1][0] # m
    innerRadius = wallPathPolar[0][0] # m

    wallPathCart = tuple(polarToCartesian(r, thetaRad) for r, thetaRad in wallPathPolar) # xy [m]
    annulusArea = np.pi * (outerRadius**2 - innerRadius**2) # m^2
    idealBucketWallArea = idealAreaRatio * annulusArea # m^2

    bucketWallArea = pathLength(wallPathCart) * bucketWallWidth # m^2

    return round(idealBucketWallArea / bucketWallArea) - 1


def getAutoStallTorquePerMetre(wallPathPolar, bucketWallWidth, plotEveryResult=False, plotBestResult=False, numberOfRotations=5):
    """
    Calculates the stall torque of a waterwheel per metre of axial width.

    Automatically tries different rotations to find the one with highest stall torque

    # Parameters:
        - wallPathPolar (array-like): The shape of the bucket, defined as an array-like
        of points, where each point is in the form (radius from centre, angle 
        anti-clockwise from +x axis in radians)
        
        - numberOfBuckets (int): Number of buckets in the wheel
        
        - plotResult (bool): Whether to draw the waterwheel with filled buckets in matplotlib
        
        - initialRotationRad (float): Add initial rotation to the wheel
    
        """

    numberOfBuckets = getIdealNumberOfBuckets(wallPathPolar, bucketWallWidth)

    initialRotationArr = np.linspace(0, 2*np.pi, numberOfRotations)
    maxStallTorquePM = 0.0
    bestTheta = initialRotationArr[0]

    for theta in initialRotationArr:

        stallTorquePM = getStallTorquePerMetre(wallPathPolar, numberOfBuckets, plotEveryResult, theta)

        if stallTorquePM > maxStallTorquePM:
            maxStallTorquePM = stallTorquePM
            bestTheta = theta

    if plotBestResult:
        getStallTorquePerMetre(wallPathPolar, numberOfBuckets, True, bestTheta)

    return maxStallTorquePM

def getStallTorquePerMetre(wallPathPolar, numberOfBuckets: int, plotResult=False, initialRotationRad=0.0):
    """
    Calculates the stall torque of a waterwheel per metre of axial width.

    # Parameters:
        - wallPathPolar (array-like): The shape of the bucket, defined as an array-like
        of points, where each point is in the form (radius from centre, angle 
        anti-clockwise from +x axis in radians)
        
        - numberOfBuckets (int): Number of buckets in the wheel
        
        - plotResult (bool): Whether to draw the waterwheel with filled buckets in matplotlib
        
        - initialRotationRad (float): Add initial rotation to the wheel

    """

    # check that first angle is 0
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
        wheelThetaRad += initialRotationRad 

        for j , (r, bucketThetaRad) in enumerate(wallPathPolar):

            # polar
            bucketPointsPolar[i][j][0] = r
            bucketPointsPolar[i][j][1] = bucketThetaRad + wheelThetaRad

            # cartesian
            bucketPointsCart[i][j] = polarToCartesian(r, bucketThetaRad + wheelThetaRad)


    # fill buckets
    waterPolyList: list[Any] = [None] * numberOfBuckets
    waterPolyCentroidList: list[Any] = [None] * numberOfBuckets
    bucketAreas = np.zeros((numberOfBuckets))
    stallTorqueAccumulator = 0 #Nm/m
    stallTorqueContributions = np.zeros((numberOfBuckets)) #Nm/m
    for i in range(numberOfBuckets):
        wheelThetaRad = i * np.pi * 2 / numberOfBuckets

        # Skip buckets that would create anti-clockwise torque (left side of wheel)
        if np.pi/2 < wheelThetaRad+initialRotationRad < 3*np.pi/2:
            continue

        bucketWallPathRightCart = bucketPointsCart[i]
        bucketWallPathLeftCart = bucketPointsCart[(i + 1) % numberOfBuckets]

        # Process geometry via Shapely engine
        water_poly, water_area, centroid = get_bucket_water_polygon(
            bucketWallPathRightCart, 
            bucketWallPathLeftCart
        )



        if water_poly is None or centroid[0] <= 0: # type: ignore
            continue

        # Calculate stall torque contribution
        stallTorqueContribution = water_area * 1000 * 9.81 * centroid[0] # type: ignore
        stallTorqueAccumulator += stallTorqueContribution

        # Convert shapely polygon back to numpy for matplotlib plotting
        waterPolyList[i] = np.array(water_poly.exterior.coords) # type: ignore
        waterPolyCentroidList[i] = np.array(centroid)
        bucketAreas[i] = water_area
        stallTorqueContributions[i] = stallTorqueContribution

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

            showText = True
            if showText:
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

if __name__ == "__main__":


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
                    (0.35, 0.1),
                    (0.40, 0.2),
                    (0.45, 0.3),
                    (0.50, 0.4),)

    wallPathPolar2 = ((0.20, 0),
                    (0.25, -0.1),
                    (0.30, 0),
                    (0.50, 0.40))

    torquePerMetre = getStallTorquePerMetre(wallPathPolar1, 16, True)


    print(f"Stall torque per metre of axial width = {round(torquePerMetre,2)} Nm/m")
