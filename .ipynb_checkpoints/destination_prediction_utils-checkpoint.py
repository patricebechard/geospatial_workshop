import os
import sys
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def convert_dset_to_json(dataset):
    
    dataset_json = []
    for elem in dataset:
        dataset_json.append({})
        dataset_json[-1]["input"] = elem[0]
        dataset_json[-1]["target"] = elem[1]
    
    return dataset_json

def download_data(download_url="https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip",
                  folder_name="geolife_trajectories"):
    """
    Parameters
    ----------
    download_url : string
        url of Geolife dataset
    folder_name : string
        name of folder where the data will be stored
    """
    
    import requests
    import zipfile
    import io
    
    # download data if does not exist yet
    if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
        r = requests.get(download_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        os.rename("Geolife Trajectories 1.3", folder_name)    

def show_trip_length_distribution(folder_name="geolife_trajectories", 
                                  n_bins=100, lower=50, upper=2000):
    """
    Parameters
    ----------
    folder_name : string
        name of folder where the data is stored
    lower : int
        shortest trip length shown by histogram
    upper : int
        longest trip length shown by histogram
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # create list of trip lengths
    trip_lengths = []
    # loop over users
    for user in os.listdir(os.path.join(os.getcwd(), folder_name, "Data")):
        user_path = os.path.join(os.getcwd(), folder_name, "Data", user)
        # loop over trips of each user
        for trip in os.listdir(os.path.join(user_path, "Trajectory")):
            trip_path = os.path.join(user_path, "Trajectory", trip)
            data = pd.read_csv(trip_path, header=None, skiprows=6)
            trip_lengths.append(len(data))

    # create histogram to show trip length distribution
    plt.hist(trip_lengths, bins=n_bins, range=(lower, upper))
    plt.show()
    
def create_datasets(folder_name="geolife_trajectories", lower=50, upper=2000,
                    train_frac=0.8, valid_frac=0.1):
    """
    Parameters
    ----------
    folder_name : string
        name of folder where the data is stored
    lower : int
        Minimum length of kept trips
    upper : int
        Maximum length of kept trips
    train_frac : float
        Proportion of whole dataset dedicated to training set
    valid_frac : float
        Proportion of whole dataset dedicated to validation set
    """
    import pandas as pd
    import numpy as np
    import json
    
    assert (train_frac + valid_frac) <= 1.

    # store each trip with proper length here
    combined_dset = []

    # loop over users
    for user in os.listdir(os.path.join(os.getcwd(), folder_name, "Data")):
        user_path = os.path.join(os.getcwd(), folder_name, "Data", user)
        # loop over trips of each user
        for trip in os.listdir(os.path.join(user_path, "Trajectory")):
            trip_path = os.path.join(user_path, "Trajectory", trip)
            data = pd.read_csv(trip_path, header=None, skiprows=6)
            
            # check if length of trips satisfies our criteria
            if len(data) < lower or len(data) > upper:
                continue

            latitudes = data.iloc[:,0].values
            longitudes = data.iloc[:,1].values

            # store data so that full trip is trajectory, last point is target for supervised learning
            trajectory = [[latitudes[i], longitudes[i]] for i in range(len(data))]
            target = [latitudes[-1], longitudes[-1]]

            combined_dset.append((trajectory, target))
    
    # shuffle dataset
    np.random.shuffle(combined_dset)
    
    # split into train, valid, test set
    train_cutoff = int(len(combined_dset) * train_frac)
    valid_cutoff = int(len(combined_dset) * valid_frac) + train_cutoff
    train_dset = combined_dset[:train_cutoff]
    valid_dset = combined_dset[train_cutoff:valid_cutoff]
    test_dset = combined_dset[valid_cutoff:]
    
    # saving training set
    with open(os.path.join(os.getcwd(), folder_name, 'train.json'), 'w') as json_file:
        json.dump(convert_dset_to_json(train_dset), json_file)

    # saving valid set
    with open(os.path.join(os.getcwd(), folder_name, 'valid.json'), 'w') as json_file:
        json.dump(convert_dset_to_json(valid_dset), json_file)

    # saving valid set
    with open(os.path.join(os.getcwd(), folder_name, 'test.json'), 'w') as json_file:
        json.dump(convert_dset_to_json(test_dset), json_file)
    
if __name__ == "__main__":
    
    #download_data()
    
    #show_trip_length_distribution()
    
    create_datasets()