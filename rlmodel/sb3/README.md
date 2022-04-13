# Content

## SAC and TD3
These notebooks use the [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) implementation of the `RL` algorithms and train agents based on the ground truth data from the NRP simulation to solve the reaching task. The notebooks give the possibility to play around with a large number of hyperparameters, like the usage of HER, the number of actuated joints, action- and state-space definition, threshold scheduling, and additional network parameters. Our best performing `SAC` and `TD3` models are located in the [saved_models](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/tree/master/rlmodel/sb3/saved_models) folder and can be evaluated using the notebooks.

The table below contains the evaluation of our TD3 and SAC models. The threshold is the minimum proximity of the robot end effector to the target in order to consider an episode successful. The actuated joints show which joints could be used by the agent to solve the reaching task.
|| Actuated Joints | Reward Type | HER | Average Distance (m) | <br>20 cm<sup>a</sup> |<br>15 cm<sup>a</sup> |<br>10 cm<sup>a</sup> | Success Rate (–)<br>7 cm<sup>a</sup>  |<br>5 cm<sup>a</sup> |<br>3 cm<sup>a</sup> | Training length<sup>b</sup> |
|:-----:|:---------:|:--------:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| TD3 | 1-3<br>1-3<br>1-3<br>1-3<br>1-3, 5<br>1-6 | dense<br>dense<br>sparse<br>sparse<br>dense<br>dense | no<br>yes<br>no<br>yes<br>yes<br>yes | 0.057<br>0.052<br>0.074<br>0.062<br>0.024<br>0.046 | 0.999<br>0.998<br>0.981<br>1.000<br>N/A<br>N/A | 0.978<br>0.985<br>0.907<br>0.975<br>N/A<br>N/A | 0.872<br>0.890<br>0.799<br>0.849<br>0.982<br>0.866 | N/A<br>N/A<br>N/A<br>N/A<br>0.958<br>0.806 | 0.542<br>0.658<br>0.381<br>0.472<br>0.924<br>0.744 | N/A<br>N/A<br>N/A<br>N/A<br>0.766<br>0.534 | 10000<br>10000<br>10000<br>10000<br>30000<br>60000 |
| SAC | 1-3<br>1-3<br>1-3<br>1-3<br>1-3, 5<br>1-3, 5<br>1-3, 5<br>1-3, 5<br>1-6 | dense<br>dense<br>sparse<br>sparse<br>dense<br>dense<br>sparse<br>sparse<br>sparse | no<br>yes<br>no<br>yes<br>no<br>yes<br>no<br>yes<br>yes | 0.070<br>0.091<br>0.054<br>0.109<br>0.047<br>0.086<br>0.029<br>0.042<br>0.025 | 0.986<br>0.988<br>0.988<br>0.866<br>1.000<br>1.000<br>N/A<br>N/A<br>N/A | 0.936<br>0.884<br>0.940<br>0.788<br>0.984<br>0.992<br>N/A<br>N/A<br>N/A | 0.758<br>0.568<br>0.868<br>0.612<br>0.948<br>0.840<br>0.966<br>0.924<br>1.000 | N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>0.932<br>0.864<br>0.978 | 0.454<br>0.252<br>0.646<br>0.342<br>0.760<br>0.688<br>0.866<br>0.742<br>0.856 | N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>0.600<br>0.376<br>0.716 | 5000<br>5000<br>5000<br>5000<br>4000<br>4500<br>4000<br>4500<br>13000 |

<sup>a</sup> Evaluation threshold.  
<sup>b</sup> The length of the training process is given in the number of episodes.


## EVALUATION
This notebook enables fast evaluation of the best performing models. Trained models are located in the saved_models folder. 

## SAC_IMG

`SAC_IMG.ipynb` trains a robot arm to reach a randomly spawning cylinder based on coordinates extracted from image data. Reading the image data from a .pi file, a PyTorch custom dataset class ([customDataset.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/blob/master/rlmodel/image_processing/src/custom_dataset.py)) processes and retrieves the cylinder coordinates from the images.

Retrieving the cylinder coordinates

* A camera mounted above the table captures images of the whole table surface.
* A pipeline then processes the images and extracts the location of the cylinders via subtraction, thresholding, and dilation.
The resulting dataset is then passed to and handled by a train and a test data loader.

#### CNN Policy provided by Stable Baselines3

Additionally, [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations) provides a CNN policy to deal with image input. Since in the reaching task the observation contains both image and joint angle values, this policy must handle multiple inputs by using `Dict` Gym space. In this case however, HER cannot be implemented because Stable Baselines3 doesn't support nested dictionary for now. (HER also requires a `Dict` Gym space)

This CNN policy is tested in SAC algorithm. Although the robot arm is able to reach the cylinder closer to receive more reward, the final performance is not satisfactory and still subject to development.

## TD3_IMG

`TD3_IMG.ipynb` does the same as `SAC_IMG.ipynb`, but uses the TD3 algorithm. Additionally, it does not train a model from scratch. Instead, it loads the best performing TD3 model from the saved model folder, continuous the model's training, and then evaluates it using the coordinates extracted from the images. 
