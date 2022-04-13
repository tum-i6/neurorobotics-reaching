# How to use image API and image datasets

## Image Acquisition

To interact with the cameras - acquire images - solely use the functions from `experiment_api_wrapper.py`

For a more detailed use of `experiment_api_wrapper`, please see image_processing.ipynb

## Image Datasets
The image datasets are located under: 
```
/tum_nrp/rlmodel/image_processing/data
```

There are currently three datasets. The datasets are saved using:
```
>>> torch.save(dataset, "${PATH_TO_DATASET}")
```

To use the datasets, use:
```
>>> import torch
>>> torch.load("${PATH_TO_DATASET}")
```

Each dataset is saved in the form of 
```
List[image: torch.Tensor(), cylinder_position: list()]
```

`no.cylinder.pt` contains 100 data without cylinder and `training_data.pt` contains 2000 data. 


## Image Processing

### autoen.ipynb

The notebook `autoen.ipynb` loads images from a `<name>.pi` file and extracts their latent space representation via an autoencoder.

###  cylinder_loc.ipynb


The notebook `cylinder_loc.ipynb` shows how the image data, written to a `<name>.pi` file, can be handled and manipulated.

* Create PyTorch data set via [customDataset.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/blob/master/rlmodel/image_processing/src/custom_dataset.py)
  * Manually extracts the cylinder location
  * Provide a function to iterate the extracted location, ground truth, and raw image data
* Load and manipulate the data using PyTorch transforms and data loaders
* Visualize the data
* Convert the extracted cylinder location from image to simulation coordinate frame via the custom transformation class [transform_nonzeros.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/blob/master/rlmodel/image_processing/src/transform_nonzeros.py)
* Calculate the error margin of the extracted cylinder location vs. the ground truth

### pca_dim_reduction.ipynb

The notebook `pca_dim_reduction.ipynb` contains our preliminary approach to extract a latent space representation from the images without using machine learning. This latent space representation has a decreased dimensionality and can be used as input to the `RL` agent in the _learning from images_ task. This idea has not yet been thoroughly investigated.

### src/customDataset.py
[CustomDataset.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/blob/master/rlmodel/image_processing/src/custom_dataset.py) is an implementation of the PyTorch Dataset class. The class implementation loads the image data from .pi files. Each file is masked and thresholded. The remaining blob representing the cylinder is then dilated. The class returns a binary matrix, with all nonzero values representing the cylinder, as x value and the ground truth location of the cylinder as the y value.

### src/transform_nonzeros.py

The [customDataset.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/blob/master/rlmodel/image_processing/src/custom_dataset.py) data loader returns a binary matrix that marks the location of the cylinder with nonzero values.
However, we only require the x and y value of the location of the cylinder. Further, we want to transfer the detected x and y values from the image coordinate frame to the simulation coordinate frame. [Transform_nonzeros.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/blob/master/rlmodel/image_processing/src/transform_nonzeros.py) holds a custom transform class that converts the extracted cylinder location from image frame to simulation frame.
