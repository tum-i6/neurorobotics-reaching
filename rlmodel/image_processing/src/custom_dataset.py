# Tensor Operations
import torch

# Dataset creation and handling
import os
import glob
from torch.utils.data import Dataset
from skimage import io

class CMLRPiDataset(Dataset):
    def __init__(self, root_dir, data_pi_name, mask_pi_name=None, threshold=None, transform=None, add_img=False):
        """ Initializes the dataset.
        
            Args:
                root_dir: path to root directory
                data_pi_name: name of the .pi data file
                mask_pi_name: name of the .pi masking file - image used to mask out the background
                threshold: if set an image get split at the thrashold in to zero and one values 
                transform: a pytorch transform compose
                add_img: add the image as third value to each item
                
        """
        
        # set class variables
        self.transform = transform
        self.mask_pi_name = mask_pi_name
        self.threshold = threshold
        self.add_img = add_img
        
        # retrive training images
        self.trainings_data = torch.load(os.path.join(root_dir, data_pi_name))
        # retrive masking images
        no_cylinder = torch.load(os.path.join(root_dir, mask_pi_name))
        self.data_no_cyl = [img for img, label in no_cylinder]
        # create a masking mask based by averaging the images with out cylinder
        self.avg_image = torch.stack(self.data_no_cyl, dim=3).sum(dim=3)/len(no_cylinder)
    
    def __getitem__(self, index):
        """ Returns a sample and its corresponding ground truth label.
        
            Args:
                index: The index of the sample
                
            Return:
                The class returns a binary matrix, with all nonzero values representing the cylinder, 
                as x value and the ground truth location of the cylinder as the y value.
        """
        # substract the mask from the image if it exists
        x = (self.trainings_data[index][0] - self.avg_image)*1.0
        
        # thrashold the image and convert to single channel
        x[x < self.threshold] = 0
        x = torch.sum(x, dim=-1)
        x[x >= self.threshold] = 1
        x = x.unsqueeze(-1)
        
        # permute for transform operations
        x = x.permute(2, 0, 1)
        if self.transform:
            x = self.transform(x)
        if not self.add_img:
            return x, torch.tensor(self.trainings_data[index][1])
        else:
            return x, torch.tensor(self.trainings_data[index][1]), self.trainings_data[index][0].permute(2, 0, 1)
        
    def __len__(self):
        return len(self.trainings_data)