# Stall torque simulations docs

This repo contains scripts that help to design water wheel bucket shapes by providing functions to calculate the stall torque of a given bucket geometry

stallTorqueCalc.py is where these functions can be found. The most important one is

```
def getAutoStallTorquePerMetre(wallPathPolar, bucketWallWidth, plotEveryResult=False, plotBestResult=False, numberOfRotations=5):
    """
    
    Calculates the stall torque of a waterwheel per metre of axial width.

    Automatically tries different rotations to find the one with highest stall torque

    Also Automatically calculates the ideal number of buckets and prints to console

    # Parameters:
        - wallPathPolar (array-like): The shape of the bucket, defined as an array-like
        of points, where each point is in the form (radius from centre, angle 
        anti-clockwise from +x axis in radians)
        
        - numberOfBuckets (int): Number of buckets in the wheel
        
        - plotResult (bool): Whether to draw the waterwheel with filled buckets in matplotlib
        
        - initialRotationRad (float): Add initial rotation to the wheel
    """
```

The "tester" scripts are showcases of how you can vary the geometry in a controlled experiment to find the ideal parameters.



