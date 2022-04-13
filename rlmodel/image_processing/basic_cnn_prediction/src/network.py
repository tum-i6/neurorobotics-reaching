import torch
import torch.nn as nn

class Model(nn.Module):
    '''Standard CNN network for image processing
        
        Args:
            x: image of shape (3x120x120)
            params: parameters as dictionary        
    '''
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.model = nn.Sequential(
            nn.Conv2d(3, self.params["hidden_channel"], kernel_size=3, padding=1), 
            nn.GELU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(self.params["hidden_channel"], self.params["hidden_channel"], kernel_size=3, padding=1), 
            nn.GELU(), 
            nn.MaxPool2d(2), 
            nn.Flatten(), 
            nn.Linear(30*30*self.params["hidden_channel"], self.params["hidden_layer"]), 
            nn.GELU(), 
            nn.Linear(self.params["hidden_layer"], 2)
        )

    def forward(self, x):
        return self.model(x).float()
    
class PretrainedModel(nn.Module):
    '''Used for pre-train the conv layers on cifer data, then later loaded into class "Model". 
        Args: 
            x: img
    '''
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.model = nn.Sequential(
            nn.Conv2d(3, self.params["hidden_channel"], kernel_size=3, padding=1), 
            nn.GELU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(self.params["hidden_channel"], self.params["hidden_channel"], kernel_size=3, padding=1), 
            nn.GELU(), 
            nn.MaxPool2d(2), 
            nn.Flatten(), 
            nn.Linear(8*8*self.params["hidden_channel"], self.params["hidden_layer"]), 
            nn.GELU(), 
            nn.Linear(self.params["hidden_layer"], 10)
        )

    def forward(self, x):
        return self.model(x).float()