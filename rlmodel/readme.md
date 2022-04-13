## Using the RL container

First, make sure that the `NRP` and `RL` containers are running ([hint](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask#starting-and-stopping-the-nrp)).

Start the simulation either using the web frontend or via VirtualCoach as described [here](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask#accessing-the-nrp-web-frontend).

Don't forget to establish the `gRPC` communication between the `NRP` backend and the `RL` container as described [here](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/tree/master/grpc#starting-the-grpc-server).

Enter the `RL` container running:
```bash
docker exec -it nrp-docker_rl_1 /bin/bash
```

Start a jupyter notebook:
```bash
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

Connect to the notebook via the link depicted after running the command above.

Open Tensorboard GUI (don't forget to change the path below) via the link depicted after running the command:
```bash
tensorboard --logdir path/to/tensorboard/logs --host 0.0.0.0 --port 6006
```

