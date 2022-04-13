# tensor Operations
import torch

class non_zeros(object):
    """Custom transform class that converts the extracted cylinder location from image frame to simulation frame.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    # transformation of the y value in the image coordinat frame to the y value in the simulation coordinate frame
    def cyl_y(self, x_pixel, t_x_small= -0.44, t_x_large=0.48):
        return t_x_large - ((t_x_large+abs(t_x_small))*x_pixel)/self.height

    # transformation of the x value in the image coordinat frame to the x value in the simulation coordinate frame
    def cyl_x(self, y_pixel, t_y_small=-0.48, t_y_large=0.44):
        return t_y_large - ((t_y_large+abs(t_y_small))* y_pixel)/self.width 

        
    def __call__(self, tensor):
        """ Detects the nonzero blob from the substracted and thresholded image tensor and transforms its
            location to the simulation coordinate frame.
            
            Args:
                tensor: The processed image 
            
            Return:
                The x and y value of the detected cylinder in the simulation coordinate frame.
        """
        cyl_loc = torch.mean(torch.Tensor.float(torch.nonzero(tensor)),dim=0)
        return torch.tensor((self.cyl_x(cyl_loc[1]),self.cyl_y(cyl_loc[2]),1.12),dtype=torch.float32)
    
    def __repr__(self):
        return self.__class__.__name__ 