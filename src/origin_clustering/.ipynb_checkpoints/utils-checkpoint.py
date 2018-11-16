import os
import sys

def create_datasets(folder_name="geolife_trajectories",
                    long_lower=116.25, long_upper=116.5,
                    lat_lower=39.85, lat_upper=40.1):
    """
    Parameters
    ----------
    folder_name : string
        name of folder where the data is stored
    long_lower : int
        minimum longitude kept
    long_upper : int
        maximum longitude kept
    lat_lower : int
        minimum latitude kept
    lat_upper : int
        maximum latitude kept
    """

    import pandas as pd
    import numpy as np
    import json
    
    # store each trip with proper length here
    dset = []

    # loop over users
    print("Retrieving origins and destinations from all trips ...")
    for user in os.listdir(os.path.join(os.getcwd(), folder_name, "Data")):
        user_path = os.path.join(os.getcwd(), folder_name, "Data", user)
        out_of_bounds=False
        # loop over trips of each user
        for trip in os.listdir(os.path.join(user_path, "Trajectory")):
            trip_path = os.path.join(user_path, "Trajectory", trip)
            data = pd.read_csv(trip_path, header=None, skiprows=6)

            # we only keep beginning and end points
            for point in [data.iloc[0,:2].values, data.iloc[-1,:2].values]:
            #for point in [data.iloc[0,:2].values]:
                if point[0] < lat_lower or point[0] > lat_upper or point[1] < long_lower or point[1] > long_upper:
                    # out of bounds
                    continue
                else:
                    dset.append(point)

    print("Saving the dataset created...")
    np.savetxt(os.path.join(os.getcwd(), folder_name, "origins.csv"), dset)
