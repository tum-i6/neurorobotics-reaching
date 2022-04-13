# How to use GRPC communication
## Preperation Steps
Change to TUM_NRP_DIR/nrp-docker and run.
```bash
sudo docker-compose -f run-nrp.yml up -d
```
Start the backend bash in the terminal.
```bash
sudo docker-compose -f run-nrp.yml exec backend1 bash
```
Start the rl bash in another terminal.
```bash
sudo docker-compose -f run-nrp.yml exec rl bash
```

If the Backend and the RL bashes are running, start the frontend which is accessible at [http://localhost:9000/#/esv-private?dev](http://localhost:9000/#/esv-private?dev).

(Make sure to select backend1.)

## Starting the GRPC-Server
First of all start the communication server in the Backend bash.
```bash
python3 /tum_nrp/grpc/python/communication/communication_server.py
```
The server is ready when the GRPC-server-port is printed to the bash.
```bash
GRPC-server-port: 172.18.0.x:50051
```

## Using the GRPC-Client
The GRPC-client is used from the RL container.

Code of the GRPC-client class is accessible with
```bash
gedit /grpc/python/communication/communication_client.py
```

Furthermore, it exists an experiment_api_wrapper, which provides access to all methods of the experiment api in the same manner as using the experiment_api directly.

Note:
- because of issues in the original robot.get_current_frame() method in the experiment_api, this function was not included in the experiment_api_wrapper
- because the robot.euc_distance() and cylinder.euc_distance() methods are pure methods of numpys linear algebra packages and would only produce communication overhead, these methods were not included as well

To use these methods, the RL-model just has to create an instance of the ExperimentWrapper class from the experiment_api_wrapper (just as creating a new experiment) and call the methods with the necessary parameters.

Example:
```python
# import
import experiment_api_wrapper as eaw
experiment = eaw.ExperimentWrapper()

# call some methods
experiment.setup()
experiment.robot.reset()
experiment.cylinder.is_stable()
```
 
If you want to test the GRPC-connection and the experiment-wrapper you can use
```bash
cd rlmodel/
python3 dummy_model.py
```
inside the RL container.

## Deprecated Versions
Direct use of the GRPCClient class from communication_client.py is now depracted, but still supported.
If you use these methods, please make sure to migrate to the new more intuitive ExperimentWrapper class from experiment_api_wrapper.py
