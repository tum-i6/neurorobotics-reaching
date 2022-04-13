# Data Generator
This is the python code that we used to generate and check the image dataset, on which our image models and other computer vision methods are based. 

## data_generator
`data_generator.py` is used to generate the image dataset. The size of the dataset and the saving directory is defined in the python file. The dataset is generated in this form: 
```
List[image: torch.tensor((120, 120, 3)), cylinder_position: float]
```
The sleep time in `data_generator.py` is included because the images need to be taken with the robot arm in its intial position. The sleep time is introduced to ensure that the `setup()` function is fully conducted.

### data_check
`data_check.py` is used to check if the data are generated properly.