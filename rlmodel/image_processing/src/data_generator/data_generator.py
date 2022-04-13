import sys
sys.path.append("../../../../grpc/python/communication")

sys.path.insert(1, '/tum_nrp/grpc/python/communication')
import experiment_api_wrapper as eaw
import torch
import time
sys.path.insert(1,'/tum_nrp/rlmodel/image_processing/src/data_generator')

#Set up experiment
exp = eaw.ExperimentWrapper()
exp.setup()
exp.cylinder.random_reset()
time.sleep(0.5)

#Define params
sleep_time = 0.3
data_size = 10000
start_new_data = False

#Load data and sve them
print("Start to load data.")
training_data = []
for i in range(data_size):
    exp.cylinder.random_reset()
    time.sleep(sleep_time)
    training_data.append([
        torch.tensor(exp.camera.get_image()), 
        exp.cylinder.get_position()
    ])
    if i%100==0:
        print(f'{i+1} data have been loaded')
print("Data is loaded.")
torch.save(training_data, '../../data/training_data_10k.pt')
print("Data is saved.")

#Reload and check data
print("Re-loading data")
new_data = torch.load('../../data/training_data_10k.pt')
for i in range(data_size):
    print(new_data[i][1])
