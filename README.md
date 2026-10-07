# Program Purpose
To identify and localize objects at waypoints along the route determined by a GraphNav map, by commanding Spot to take photos of its surroundings at every waypoints,
running YOLO inference, and matching the identified classes to the user's desired target object. With a success, the program will notify the user which waypoint the object was located at, otherwise, it will report failure at the end of the route.

# Dependencies
The program uses various bosdyn libraries, numpy, argparse, sys, time, and cv2.

# How to run
The program requires the complete upload filepath of the downloaded graph map, with the name 'downloaded_graph'. It also requires a target object. The default confidence threshold is 0.65, but should the user desire to change it, it can be done through the command line.

# Acknowledgements
Sections of code were recycled from the Boston Dynamics SDK.
