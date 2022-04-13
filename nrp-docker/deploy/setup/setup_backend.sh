# General NRP Setup
echo "Setting up NRP Backend"
echo "127.0.0.1 $(uname -n)" | sudo tee --append /etc/hosts

source /home/bbpnrsoa/nrp/src/GazeboRosPackages/devel/setup.sh

/bin/sed -e "s/localhost:9000/$EXTERNAL_FRONTEND_IP:9000/" -i /home/bbpnrsoa/nrp/src/ExDBackend/hbp_nrp_commons/hbp_nrp_commons/workspace/Settings.py
/bin/sed -e "s/localhost:9000/$EXTERNAL_FRONTEND_IP:9000/" -i /home/bbpnrsoa/nrp/src/VirtualCoach/hbp_nrp_virtual_coach/hbp_nrp_virtual_coach/config.json

# Apply patches
/bin/sed -e "s/24 \* 60 \* 60/365 \* 24 \* 60 \* 60/" -i /home/bbpnrsoa/nrp/src/ExDBackend/hbp_nrp_commons/hbp_nrp_commons/workspace/Settings.py

# Fix Docker Image issues
mkdir -p /home/bbpnrsoa/.gazebo/models

cd $HOME/nrp/src
source $HOME/.opt/platform_venv/bin/activate
pyxbgen -u Experiments/bibi_configuration.xsd -m bibi_api_gen
pyxbgen -u Experiments/ExDConfFile.xsd -m exp_conf_api_gen
pyxbgen -u Models/environment_model_configuration.xsd -m environment_conf_api_gen
pyxbgen -u Models/robot_model_configuration.xsd -m robot_conf_api_gen
deactivate

gen_file_path=$HBP/ExDBackend/hbp_nrp_commons/hbp_nrp_commons/generated
filepaths=$HOME/nrp/src
sudo cp $filepaths/bibi_api_gen.py $gen_file_path
sudo cp $filepaths/exp_conf_api_gen.py $gen_file_path
sudo cp $filepaths/_sc.py $gen_file_path
sudo cp $filepaths/robot_conf_api_gen.py $gen_file_path
sudo cp $filepaths/environment_conf_api_gen.py $gen_file_path

# Installation of Custom Experiments
sudo cp /tum_nrp/experiment/experiment_api.py $HOME

# Add your scripts and commands here
sudo python -m pip install grpcio
sudo python -m pip install --upgrade protobuf

# Start NRP Backend
echo "Starting NRP Backend"
/home/bbpnrsoa/nrp/src/user-scripts/rendering_mode cpu
/home/bbpnrsoa/nrp/src/Models/create-symlinks.sh

sudo -E /etc/init.d/supervisor stop
sudo -E /etc/init.d/supervisor start

sleep infinity

echo "done"
