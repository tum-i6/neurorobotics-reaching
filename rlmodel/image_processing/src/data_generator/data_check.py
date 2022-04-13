import torch

'''Used for checking if two consecutive data points are the same
    (This could happen when the setup() func isn't carried out completely and the image is already taken.
    One solution would be to increase sleep_time.)
'''

training_data = torch.load('../data/training_data.pt')
for i in range(len(training_data)-1):
    if training_data[i][1]==training_data[i+1][1]:
        print('There are two same data')
        break
print('Data check is done.')