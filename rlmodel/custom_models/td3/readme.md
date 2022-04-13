# Description

This is the custom implementation of the `TD3` algorithm based on this [article](https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93) and the original [paper](https://arxiv.org/abs/1802.09477). This implementation uses fully-connected neural networks to approximate the `Actor` and `Critic`. Additionally to the different experimtent configuration settings found in [main_td3.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/blob/master/rlmodel/td3_custom/main_td3.py#L16), exploration noise decay and adaptive goal threshold (based on the success rate achieved in the intermittent evaluation steps during training) is implemented to increase the performance of the agent. Tensorboard is used as a visualization and tracking toolkit for the training of models.

The best result with an agent trained with this algorithm (with dense reward function and 3 actuated joints) is:

Threshold [m] | Success rate [-] | Average distance [m]
--- | --- | ---
0.20 | 0.994 | 0.057
0.15 | 0.979 | 0.057
0.10 | 0.886 | 0.058
0.05 | 0.498 | 0.057

Maintainer: [Márton Szép](mailto:marton.szep@tum.de)

## How to run

1. Make sure to install dependencies in the containers:
```bash
$ cd TUM_NRP_DIR/nrp-docker
$ docker-compose -f run-nrp.yml up -d --build
```
2. Start the simulation in the [frontend](http://localhost:9000/#/esv-private?dev) or using `VirtualCoach`:
```bash
$ docker-compose -f run-nrp.yml exec backend1 bash
$ cd /tum_nrp/experiment
$ python
>>> import auto_sim
>>> autosim=auto_sim.AutoSim()
>>> autosim.start()
```
3. Don't forget to establish the `gRPC` communication between the `NRP` backend and the `RL` container as described [here](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/tree/master/grpc#starting-the-grpc-server). 

4. Enter RL container and run the `TD3`:
```bash
$ docker exec -it nrp-docker_rl_1 /bin/bash
$ cd rlmodel/td3_custom/
$ python3 main_td3.py --start_timesteps 10 --max_timesteps 20 --eval_freq 10 --eval_episodes 10 --proximity 0.30 --use_3_joints --comment 3joints
```
This can be used to train an agent from scratch or you could also load a model and continue the training or just evaluate it. For more information on the arguments see [main_td3.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/blob/master/rlmodel/td3_custom/main_td3.py#L16).

5. Additionally, start Tensorboard GUI to see the logs of the `TD3` algorithm:
```bash
$ tensorboard --logdir rlmodel/td3_custom/tensorboard_log --host 0.0.0.0 --port 6006
```
6. Open Tensorboard GUI via the link depicted after running the command above.