# Author: Kevin Farkas (11.Mai 2021)
#
# content:
# - simple script to run GRPC client 


import sys 
#import logging

from communication_client import GRPCClient
import experiment_api

# Create an instance of GRPCClient
client = GRPCClient()

##########################
# setup logging          #
##########################
#logging.basicConfig(filename='./logs/communication.log',
#                            filemode='a',
#                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                            datefmt='%H:%M:%S',
#                            level=logging.INFO)
#
#logger = logging.getLogger('comm_client')


##########################
# experiment_api         #
##########################

# instanciate experiment
experiment = experiment_api.Experiment()


##########################
# communication_client   #
##########################

# Create an instance of GRPCClient
client = GRPCClient()

# initialize episode variable
episode = None


# endless loop for experiments
while True:
    # request start for new experiment
    reg = client.request_start()
    print("Request for new experiment.")
    #logger.info("Request for new experiment.")
    
    # setup new experiment
    setup = experiment.setup()
    print("Setup new experiment.")
    #logger.info("Setup new experiment.")

    # report new experiment setup to model
    res = client.report_experiment_setup(setup)
    episode = True
    print("Report new experiment setup.")
    #logger.info("Report new experiment setup.")
    
    while episode: 
        # request action from model
        act = client.request_action()
        dist = experiment.execute(act[0], act[1], act[2], act[3], act[4], act[5])
        print("Request action from model.")
        #logger.info("Request action from model.")
        
        # report distance to target
        reset = client.report_distance(dist)
        print("Report distance to target.")
        #logger.info("Report distance to target.")
        
        if reset:
            # start a new episode
            episode = False
            print("Start a new episode.")
            #logger.info("Start a new episode.")
        
