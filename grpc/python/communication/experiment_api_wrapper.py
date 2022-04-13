import communication_client as cc


class ExperimentWrapper:
    """
    Wrapper for class "Experiment" from experiment_api.py
    """
    def __init__(self):
        # instanciate grpc client
        self.client = cc.GRPCClient()

        self.robot = RobotWrapper(self.client)
        self.cylinder = CylinderWrapper(self.client)
        self.camera = CameraWrapper(self.client)
        
    def setup(self):
        return self.client.setup()
        
    def execute(self, action):
        return self.client.execute(action)
        
    def distance_robot_cylinder(self):
        return self.client.distance_robot_cylinder()


class RobotWrapper:
    """
    Wrapper for class "Robot" from experiment_api.py
    """
    def __init__(self, client):
        # get client
        self.client = client
        
    def reset(self):
        self.client.robot_reset()
        
    def get_joint_states(self):
        return self.client.get_robot_joint_states()

    def get_position(self):
        return self.client.get_robot_position()
    
    def act(self, action):
        return self.client.robot_act(action)
        
    def is_stable(self):
        return self.client.robot_is_stable()
        
    def check_collision(self):
        return self.client.robot_check_collision()
    

class CylinderWrapper:
    """
    Wrapper for class "Cylinder" from experiment_api.py
    """
    def __init__(self, client):
        # get client
        self.client = client
    
    def reset(self):
        self.client.cylinder_reset()
        
    def random_reset(self):
        self.client.cylinder_random_reset()
        
    def get_position(self):
        return self.client.cylinder_get_position()
        
    def is_stable(self):
        return self.client.cylinder_is_stable()
        
    def is_on_ground(self):
        return self.client.cylinder_is_on_ground()
        
        
class CameraWrapper:
    """
    Wrapper for class "Camera" from experiment_api.py
    """
    def __init__(self, client):
        # get client
        self.client = client
        
    def get_image(self):
        return self.client.camera_get_image()
        
