try:
    import rospy
    from operator import itemgetter
    from std_msgs.msg import Float64
    from sensor_msgs.msg import JointState
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import CameraInfo
    from gazebo_msgs.srv import GetModelState
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import GetLinkState
    from cv_bridge import CvBridge, CvBridgeError
    import random
    import decimal
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
except:
    print("Actually you had some import problems, but this is only a warning to keep the autodoc running.")

import os

class Experiment:
    """
    Class to interact with the experiment.
    Use this to interact with the Robot and Cylinder objects.

    Examples
    --------
    >>> experiment = experiment_api.Experiment()
    >>> experiment.setup()

    Attributes
    ----------
    robot : Robot
        instance of Robot class to interact with the robot arm.
    cylinder : Cylinder
        instance of Cylinder class to interact withe the cylinder object.
    """

    def __init__(self):
        rospy.init_node('SchunkRL')

        self.table = Table()
        self.robot = Robot()
        self.cylinder = Cylinder()
        self.cameras = Cameras() #Initialize the camera class

    def setup(self):
        """ Wrapper that bundles the resetting of the robot and the random spawning of the cylinder into a single function call.

            Return:
                Position of the randomly spawned cylinder
        """
        # reset the table
        self.table.reset()
        # reset the robot
        self.robot.reset()
        # randomly spawn cylinder
        self.cylinder.random_reset()
        # return the position of the cylinder
        return self.cylinder.get_position()


    def execute(self, j1=0, j2=0, j3=0, j4=0, j5=0, j6=0):
        """ Wrapper that bundles reaching with the robot (act), checking if the cylinder and robot are stable (is_stable), 
            getting distance between the end effector and cylinder (distance_robot_cylinder) into a single function call.

            Args:
                j1-j6: Rotation of the robot joints

            Return:
                Distance between the end effector and cylinder
        """
        # move the robot to the indicated postion
        self.robot.act(j1, j2, j3, j4, j5, j6)

        #indicate if the object are stable or not
        stable_robot = False
        stable_cylinder = False

        # limit the execution time for movement of the robot arm and cylinder to 5 min
        timeout = rospy.get_time() + 5
        
        while not stable_robot or not stable_cylinder:

            try:
                # check if the robot is stable
                stable_robot = self.robot.is_stable()
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logerr("Service call failed: %s" % (e,))   
            try:
                # check if the cylinder is stable (in case that it got hit)
                stable_cylinder = self.cylinder.is_stable()
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logerr("Service call failed: %s" % (e,))   

            # limit the number of request per seconds
            rospy.sleep(1)

            # stop after 2 min
            if rospy.get_time() > timeout:
                break
        # return the distance of the end effector to the cylinder
        return self.distance_robot_cylinder()

    def distance_robot_cylinder(self):
        """ Calculates the distance between all joints and cylinder 

            Return:
                Distance between the last joint (end effector) an cylinder
        """
        # get the cylinder and robot postion
        robot_position = np.array(self.robot.get_position())
        cylinder_position = np.array(self.cylinder.get_position())

        # calculate the distance between all joints and cylinder
        distance = np.linalg.norm(robot_position - cylinder_position,axis=1)

        self.distance_joint_1_cylinder = distance[0]
        self.distance_joint_3_cylinder = distance[1]
        self.distance_joint_5_cylinder = distance[2]
        self.distance_ee_cylinder = distance[3]

        # return the distance of the end effector to the cylinder
        return self.distance_ee_cylinder

    def print_images(self):
        """Method to print the image from the camera
        """
        self.cameras.print_images()

    def get_image(self):
        """Method to get the image from the camera

           return: flattened numpy.ndarray of ONLY ONE DIMENSION (HxWxC, )
        """
        return self.cameras.get_image().reshape(-1)


class Table:
    """ Class to interact with the table on which the robot arm is deployed
    """

    def __init__(self):
        rospy.wait_for_service("gazebo/get_model_state", 20.0)
        rospy.wait_for_service("gazebo/set_model_state", 20.0)
        self.__get_pose_srv = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)
        self.__set_pose_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)

        ''' Set the inital state of the table to (0, 0, 0, 0, 0, -1, 0), 
        instead of the the state right after starting the simulation
        '''
        self.init_state = ModelState()
        self.init_state.model_name = 'robot'
        self.init_state.pose = self.__get_pose_srv('robot', 'world').pose
        self.init_state.pose.position.x = 0
        self.init_state.pose.position.y = 0
        self.init_state.pose.position.z = 0
        self.init_state.pose.orientation.x = 0
        self.init_state.pose.orientation.y = 0
        self.init_state.pose.orientation.z = -1
        self.init_state.pose.orientation.w = 0
        self.init_state.scale = self.__get_pose_srv('robot', 'world').scale

    def reset(self):
        """Resets table position to the initial state.
        """
        reset_obj = self.__set_pose_srv(self.init_state)

class Robot:
    """Class to interact with robotic arm in the simulation.
    Do not directly use this class. Instead, use the Experiment class.
    """

    def __init__(self):

        # topics for controlling the joint
        topic_1 = rospy.Publisher('/robot/arm_1_joint/cmd_pos', Float64, queue_size=20)
        topic_2 = rospy.Publisher('/robot/arm_2_joint/cmd_pos', Float64, queue_size=20)
        topic_3 = rospy.Publisher('/robot/arm_3_joint/cmd_pos', Float64, queue_size=20)
        topic_4 = rospy.Publisher('/robot/arm_4_joint/cmd_pos', Float64, queue_size=20)
        topic_5 = rospy.Publisher('/robot/arm_5_joint/cmd_pos', Float64, queue_size=20)
        topic_6 = rospy.Publisher('/robot/arm_6_joint/cmd_pos', Float64, queue_size=20)

        self.joint_topics = [topic_1, topic_2, topic_3, topic_4, topic_5, topic_6]

        # service to get position of robot's link
        rospy.wait_for_service("gazebo/get_link_state", 20.0)
        self.__get_link_srv = rospy.ServiceProxy("gazebo/get_link_state", GetLinkState)

        # Variables that hold finger states
        self.__current_state = [None]

        # Variables that hold images from virtual camera
        self.__current_image = [None]

        rospy.Subscriber("/robot/joint_states", JointState, self.__on_joint_state)
        rospy.Subscriber("virtual_camera/camera", Image, self.__receive_image)

        self.bridge = CvBridge()

    def __on_joint_state(self, msg):
        """ROS topic callback function.
        """
        timestamp = (msg.header.stamp.secs, msg.header.stamp.nsecs)
        indices = (msg.name.index('arm_1_joint'),
                   msg.name.index('arm_2_joint'),
                   msg.name.index('arm_3_joint'),
                   msg.name.index('arm_4_joint'),
                   msg.name.index('arm_5_joint'),
                   msg.name.index('arm_6_joint'))
        self.__current_state[0] = (itemgetter(*indices)(msg.position), timestamp)

    def __receive_image(self, msg):
        """ROS topic callback function.
        """
        timestamp = (msg.header.stamp.secs, msg.header.stamp.nsecs)
        # print("encoding form", msg.encoding)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")  # cv::Mat image
        except CvBridgeError as e:
            print(e)

        self.__current_image[0] = (cv_image, timestamp)

        # print("image shape", cv_image.shape)  # (512, 512, 3)
        # (rows, cols, channels) = cv_image.shape
        # if cols > 60 and rows > 60:
        #     cv2.circle(cv_image, (50, 50), 10, 255)

        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)

    def reset(self):
        """Resets robot joints to the initial state.
        """
        for topic in self.joint_topics:
            topic.publish(Float64(0.0))

    def get_joint_states(self):
        """
        Gets current state of simulation.

        Returns
        -------
        list
            joint states of the robot
        """
        # TODO: change this to service (and remove waiting times)
        # current time
        now = rospy.get_rostime() - rospy.Time(secs=0)

        # ensure that the timestamp of the joint states is greater than the current time
        while (now + rospy.Duration(0, 500000000)) > rospy.Duration(self.__current_state[0][1][0],
                                                                    self.__current_state[0][1][1]):
            rospy.sleep(0.1)
        return list(self.__current_state[0][:-1][0])

    def get_current_frame(self, resize_shape=128):
        """
        Gets the image from virtual camera.

        Returns
        -------
        image
        """
        # current time
        now = rospy.get_rostime() - rospy.Time(secs=0)

        # ensure that the timestamp of the joint states is greater than the current time
        while (now + rospy.Duration(0, 500000000)) > rospy.Duration(self.__current_state[0][1][0],
                                                                    self.__current_state[0][1][1]):
            rospy.sleep(0.1)
        # print("image shape", self.__current_image[0][0].shape)
        resize_image = cv2.resize(self.__current_image[0][0], (resize_shape, resize_shape), interpolation=cv2.INTER_AREA)
        # return self.__current_image[0][0]
        return resize_image

    def get_position(self):
        """
        Gets the current positions [x, y, z] of the joints.

        Returns
        -------
        tuple
            first entry is the position of 000 link (the 1st joint),
            second entry is the position of 002 link (the 3rd joint),
            third entry is the position of 004 link (the 5th joint) and
            the forth entry is the position of end effector (018 link).
        """
        pos_joint_1 = self.__get_link_srv(
            'robot::COL_COL_COL_VIS_VIS_VIS_VIS_VIS_VIS_VIS_VIS_VIS_hollie_real.000',
            'world').link_state.pose.position
        pos_joint_3 = self.__get_link_srv(
            'robot::COL_COL_COL_VIS_VIS_VIS_VIS_VIS_VIS_VIS_VIS_VIS_hollie_real.002',
            'world').link_state.pose.position
        pos_joint_5 = self.__get_link_srv(
            'robot::COL_COL_COL_VIS_VIS_VIS_VIS_VIS_VIS_VIS_VIS_VIS_hollie_real.004',
            'world').link_state.pose.position
        pos_ee = self.__get_link_srv(
            'robot::COL_COL_COL_VIS_VIS_VIS_VIS_VIS_VIS_VIS_VIS_VIS_hollie_real.018',
            'world').link_state.pose.position

        return (
            [pos_joint_1.x, pos_joint_1.y, pos_joint_1.z],
            [pos_joint_3.x, pos_joint_3.y, pos_joint_3.z],
            [pos_joint_5.x, pos_joint_5.y, pos_joint_5.z],
            [pos_ee.x, pos_ee.y, pos_ee.z])

    def act(self, j1, j2, j3, j4, j5, j6):
        """
        Takes an action by setting the six join positions.

        Parameters
        ----------
        ji : float
            joint value
        """
        self.joint_topics[0].publish(Float64(j1))
        self.joint_topics[1].publish(Float64(j2))
        self.joint_topics[2].publish(Float64(j3))
        self.joint_topics[3].publish(Float64(j4))
        self.joint_topics[4].publish(Float64(j5))
        self.joint_topics[5].publish(Float64(j6))

    def is_stable(self):
        """Checks if robot is stable (finished executing action).
        """
        eps = 0.001
        end_effector_old = self.get_position()[-1]
        rospy.sleep(0.1)
        end_effector_new = self.get_position()[-1]

        distance = self.euc_distance(end_effector_old, end_effector_new)

        if distance < eps:
            return True
        else:
            return False

    def check_collision(self):
        """Checks if robot suffered a collision with the table.
        """
        base, link_3, link_6, end_effector = self.get_position()
        height_check_list = [end_effector[-1], link_6[-1], link_3[-1]]
        # height_threshold = base.position.z
        height_threshold = 1.03  # height of Object:1.12, previous threshold 1.03
        for h in height_check_list:
            if h <= height_threshold:
                return True
        return False

    def euc_distance(self, pos_1, pos_2):
        """ Calculates the euclidian distance between two 2D or 3D vectors

            Args:
                pos_1: First position vector 
                pos_2: Second position vector 

            Return:
                Euclidian distance between the two passed function arguments
        """
        return np.linalg.norm(np.asarray(pos_1,dtype=np.float32)-np.asarray(pos_2,dtype=np.float32))

class Cylinder:

    def __init__(self):

        rospy.wait_for_service("gazebo/get_model_state", 20.0)
        rospy.wait_for_service("gazebo/set_model_state", 20.0)
        self.__get_pose_srv = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)
        self.__set_pose_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)

        # get the initial state of the blue object
        self.init_state = ModelState()
        self.init_state.model_name = 'BLUE_cylinder'
        self.init_state.pose = self.__get_pose_srv('BLUE_cylinder', 'world').pose
        self.init_state.scale = self.__get_pose_srv('BLUE_cylinder', 'world').scale

    def reset(self):
        """Resets cylinder position to the initial state.
        """
        reset_obj = self.__set_pose_srv(self.init_state)

    def random_reset(self):
        """Resets cylinder position to random state (on the table).
        """
        # copy the initial state
        random_state = self.init_state

        # set the x and y value to a random value limited by the size of the table
        random_state.pose.position.x = decimal.Decimal(random.randrange(-44, 48))/100
        random_state.pose.position.y = decimal.Decimal(random.randrange(-48, 44))/100
        
        # make sure that the cylinder is far enough away from the robot base
        radius = np.sqrt((float(random_state.pose.position.x) - 0.017)**2 + (float(random_state.pose.position.y)-0.4)**2)
        while (radius < 0.2):
            random_state.pose.position.x = decimal.Decimal(random.randrange(-44, 48))/100
            random_state.pose.position.y = decimal.Decimal(random.randrange(-48, 44))/100
            radius = np.sqrt((float(random_state.pose.position.x) - 0.017)**2 + (float(random_state.pose.position.y)-0.4)**2)
        
        # set the z value equal the table hight
        random_state.pose.position.z = 1.12
        random_state.pose.orientation.x = 0
        random_state.pose.orientation.y = 0
        random_state.pose.orientation.z = -1
        random_state.pose.orientation.w = 0
        reset_obj = self.__set_pose_srv(random_state)

    def get_position(self):
        """
        Gets position of the cylinder object.

        Returns
        -------
        list
            [x, y, z] coordinates of cylinder
        """
        pos = self.__get_pose_srv('BLUE_cylinder', 'world').pose.position
        return [pos.x, pos.y, pos.z]

    def is_stable(self):
        """Checks if cylinder is stable (not moving).
        """

        eps = 0.001
        pos_old = self.get_position()
        rospy.sleep(0.1)
        pos_new = self.get_position()

        distance = self.euc_distance(pos_old, pos_new)

        if distance < eps:
            return True
        else:
            return False

    def is_on_ground(self):
        """Checks if cylinder is on the ground.
        """

        eps = 0.01
        height = self.get_position()[2]
        height_init = self.init_state.pose.position.z

        if height + eps < height_init:
            return True
        else:
            return False

    def euc_distance(self, pos_1, pos_2):
        """ Calculates the euclidian distance between two 2D or 3D vectors

            Args:
                pos_1: First position vector 
                pos_2: Second position vector 

            Return:
                Euclidian distance between the two passed function arguments
        """
        return np.linalg.norm(np.asarray(pos_1,dtype=np.float32)-np.asarray(pos_2,dtype=np.float32))

class Cameras:
    """
    A ROS subscriber that subscribes to both of the cameras
    """
    def __init__(self):
        self.camerainfo_1 = None
        self.__current_image_1 = None
        self.bridge = CvBridge()
        #Create Subscriber instance for subscription
        rospy.Subscriber("/camera_1/image_raw", Image, self.__get_image_1) 

    def __get_image_1(self, msg) -> None:
        """ROS topic callback function
        """
        try:
            #The img is being passed on to OpenCV and stored as numpy array
            cv_image_1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.__current_image_1 = cv_image_1
        except CvBridgeError as e:
            print(e)
    
    def __get_camerainfo_1(self, msg):
        """ROS topic callback func
        """
        self.camerainfo_1 = msg

    def get_image(self):
        """Method used to get image
           
           return: numpy.ndarray in shape (HxWxC)
        """
        timeout = 0
        #Waiting until the image is succesfully passed on to OpenCV
        while self.__current_image_1 is None:
            rospy.sleep(0.1)
            timeout += 1
            if timeout>100:
                raise TimeoutError('The pre-defined time limit is reached. Please check the ROS topic!')
            
        return self.__current_image_1

    def get_images(self):
        """
        Method to get 2 images as a 6xHxW tensor
        Only used for multiple cameras
        """
        #TODO
        pass

    def print_images(self):
        """func to be called when you want to print the images, e.g. in a Jupyter Notebook
        """
        print(type(self.__current_image_1))
        plt.imshow(self.__current_image_1)
        print(self.__current_image_1.shape)

