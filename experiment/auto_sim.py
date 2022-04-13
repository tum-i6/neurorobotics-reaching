from pynrp.virtual_coach import VirtualCoach
import time

class AutoSim:
    """
    Class to control the NRP without having to use the web frontend.

    Attributes
    ----------
    vc: instance of VirtualCoach Python API for interacting with experiments.
        Documentation: https://neurorobotics.net/Documentation/nrp/user_manual/virtual_coach/introduction.html
        Source: https://bitbucket.org/hbpneurorobotics/virtualcoach/src/development/
    sim: instance of simulation (launched experiment)
    """
    def __init__(self, env='http://172.17.0.1:9000'): 
        """
        Attempt to connect to the NRP Web Frontend.
        
        Args:
            env: (optional) A string containing the http address of the server running the NRP. 
            Default is 'http://172.17.0.1:9000'. When running from the backend, it could also be 
            'http://frontend:9000'.
        """
        self.vc = VirtualCoach(environment=env, 
                        storage_username='nrpuser', storage_password='password')
        self.sim = None

    def clone(self, exp_id='cmlr_experiment_configuration'):
        """
        Clone experiment template given with exp_id(default CMLR). Print list of cloned experiments.

        Args:
            exp_id: (optional) Experiment template id. Default is 'cmlr_experiment_configuration'. 
                    See self.vc.print_templates() for other experiment templates.
        """
        self.vc.clone_experiment_to_storage(exp_id)
        self.vc.print_cloned_experiments()

    def start(self, exp_id='cmlr_0', server='nrp_backend_1'):
        """
        Attempt to launch a cloned experiment and start the simulation. Print state of the launched 
        simulation and running experiments.

        Args:
            exp_id: (optional) Id of the cloned experiment. Default is 'cmlr_0'. See 
                    self.print_cloned_experiments() for other available cloned experiments.
            server: (optional) The full name of the server backend to launch on. Default is 
                    'nrp_backend_1'. See self.vc.print_available_servers() for other available 
                    backend servers that are currently not running a simulation.
        """
        def launch():
            """
            Intermediate function to be able to call retry() with func parameters.
            """
            return self.vc.launch_experiment(experiment_id=exp_id, server=server)
        self.sim=retry(func=launch, limit=2)
                
        self.sim.get_state()

        self.sim.start()
        self.sim.get_state()
        self.vc.print_running_experiments()

    def stop(self):
        """
        Attempt to stop the simulation by transitioning to the "stopped" state. Print state of the
        launched simulation.
        """
        self.sim.stop()
        self.sim.get_state()

    def pause(self):
        """
        Attempt to pause the simulation by transitioning to the "paused" state. Print state of the
        launched simulation.
        """
        self.sim.pause()
        self.sim.get_state()


def retry(func, ex_type=Exception, limit=0, wait_ms=100, wait_increase_ratio=2, logger=None):
    """
    Retry a function invocation until no exception occurs.

    Args:
        func: Function to invoke.
        ex_type: (optional) Retry only if exception is subclass of this type.
        limit: (optional) Maximum number of invocation attempts (0 for unlimited)
        wait_ms: (optional) Initial wait time after each attempt in milliseconds.
        wait_increase_ratio: (optional) Increase wait period by multiplying this value after each attempt.
        logger: (optional) If not None, retry attempts will be logged to this logging.logger

    Raise:
        Last invocation exception if attempts exhausted or exception is not an instance of ex_type
    
    Return:    
        Result of first successful invocation.
    """
    attempt = 1
    while True:
        try:
            return func()
        except Exception as ex:
            if not isinstance(ex, ex_type):
                raise ex
            if 0 < limit <= attempt:
                if logger:
                    logger.warning("no more attempts")
                raise ex

            if logger:
                logger.error("failed execution attempt #%d", attempt, exc_info=ex)

            attempt += 1
            if logger:
                logger.info("waiting %d ms before attempt #%d", wait_ms, attempt)
            time.sleep(wait_ms / 1000)
            wait_ms *= wait_increase_ratio