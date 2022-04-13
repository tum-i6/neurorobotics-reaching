# Set up NRP Model Storage
echo "Setting up the NRP Model Storage"
source /home/bbpnrsoa/nrp/src/GazeboRosPackages/devel/setup.sh

echo "Updating NRP Repositories"
{ cd /home/bbpnrsoa/nrp/src/Models && git config remote.origin.fetch "+refs/heads/master*:refs/remotes/origin/master*" && sudo git checkout master18 && sudo git pull; } || { cd /home/bbpnrsoa/nrp/src && sudo find Models/ -not -name "Models" -delete && sudo git clone --progress --branch=master18 https://bitbucket.org/hbpneurorobotics/Models.git Models/; }
{ cd /home/bbpnrsoa/nrp/src/Experiments && git config remote.origin.fetch "+refs/heads/master*:refs/remotes/origin/master*" && sudo git checkout master18 && sudo git pull; } || { cd /home/bbpnrsoa/nrp/src && sudo find Experiments/ -not -name "Experiments" -delete && sudo git clone --progress --branch=master18 https://bitbucket.org/hbpneurorobotics/Experiments.git Experiments/; }

echo "Installing Custom Experiments"
# Add your scripts and commands here
sudo cp -R /tum_nrp/experiment/src/* /home/bbpnrsoa/nrp/src/Experiments/
sudo mkdir /home/bbpnrsoa/nrp/src/Models/cmlr
sudo cp -R /tum_nrp/experiment/src/cmlr/world_manipulation.sdf /home/bbpnrsoa/nrp/src/Models/cmlr
sudo cp -R /tum_nrp/experiment/src/cmlr/idle_brain.py /home/bbpnrsoa/nrp/src/Models/cmlr

echo "Setting Permissions"
sudo chown -R bbpnrsoa:bbp-ext /home/bbpnrsoa/nrp/src/Experiments && sudo chown -R bbpnrsoa:bbp-ext /home/bbpnrsoa/nrp/src/Models

echo "Generating Textures"
python /home/bbpnrsoa/nrp/src/user-scripts/generatelowrespbr.py

echo "Installation finished"