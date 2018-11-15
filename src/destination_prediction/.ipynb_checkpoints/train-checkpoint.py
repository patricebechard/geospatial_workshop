import torch
from torch import nn
from torch import optim
#from utils import device
from src.destination_prediction.utils import device, haversine_np
import numpy as np

def train(model, train_set, valid_set=None, n_epochs=20, start_lr=1e-4, trip_frac=1.,
          long_lower=116.25, long_upper=116.5, lat_lower=39.85, lat_upper=40.1,
          lr_decay=True):
    
    train_loss_tracker = []
    eval_loss_tracker = []
    
    for epoch in range(n_epochs):
        
        if lr_decay:
            if epoch < 100:
                lr = start_lr
            else:
                # learning rate decays linearly
                lr = (n_epochs - epoch)/n_epochs * start_lr
        print("learning rate : %.6f" % lr)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
                
        total_distance_error = 0
        n_pts = 0
        
        for i, (inputs, targets) in enumerate(train_set):
            
            # truncate trip to keep only beginning
            inputs = inputs[:int(trip_frac*inputs.shape[0])]
            
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            out_lat, out_long = model(inputs)

            # re-scale output from [0,1] to the pre-defined bbox
            out_lat = (out_lat * (lat_upper - lat_lower)) / 2 + lat_lower
            out_long = (out_long * (long_upper - long_lower)) / 2 + long_lower

            loss = criterion(out_lat.squeeze(), targets[:,0])
            loss += criterion(out_lat.squeeze(), targets[:,1])
            
            loss.backward()
            optimizer.step()

            # compute mean distance error (km)
            n_pts += inputs.shape[1]
            
            out_lat = out_lat.squeeze().data.to('cpu').numpy()
            out_long = out_long.squeeze().data.to('cpu').numpy()
            tgt_lat = targets[:,0].to('cpu').numpy()
            tgt_long = targets[:,1].to('cpu').numpy()
            
            total_distance_error += np.sum(haversine_np(out_long, out_lat, tgt_long, tgt_lat))
        
        mean_distance_error = total_distance_error / n_pts
        
        mean_valid_distance_error = evaluate(model, valid_set)
            
        print("Epoch %d ----- mean distance error : %.3f ----- mean valid distance error : %.3f" % (epoch, mean_distance_error, mean_valid_distance_error))
            
        
def evaluate(model, valid_set, long_lower=116.25, long_upper=116.5, lat_lower=39.85, lat_upper=40.1):
        
    total_distance_error = 0
    n_pts = 0    
    
    for i, (inputs, targets) in enumerate(valid_set):
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        out_lat, out_long = model(inputs)

        # re-scale output from [0,1] to the pre-defined bbox
        out_lat = (out_lat * (lat_upper - lat_lower)) / 2 + lat_lower
        out_long = (out_long * (long_upper - long_lower)) / 2 + long_lower

        # compute mean distance error (km)
        n_pts += inputs.shape[1]

        out_lat = out_lat.squeeze().data.to('cpu').numpy()
        out_long = out_long.squeeze().data.to('cpu').numpy()
        tgt_lat = targets[:,0].to('cpu').numpy()
        tgt_long = targets[:,1].to('cpu').numpy()
        
        total_distance_error += np.sum(haversine_np(out_long, out_lat, tgt_long, tgt_lat))

    mean_distance_error = total_distance_error / n_pts   
    return mean_distance_error


if __name__ == "__main__":
    
    from model import DestinationLSTM
    from utils import device
    
    seq_len = 2000
    batch_size = 64
    input_size=2
    
    inputs = torch.ones((seq_len, batch_size, input_size)).to(device)
    targets = torch.ones((batch_size, input_size))
    
    model = DestinationLSTM().to(device)
    
    train(model, train_set=[(inputs, targets)], n_epochs=100)
    
    evaluate(model, valid_set=[(inputs, targets)])    