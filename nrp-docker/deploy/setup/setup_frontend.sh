# Setting up NRP Frontend
echo "Setting up NRP Frontend"
/bin/sed -e "s/localhost:9000/$EXTERNAL_FRONTEND_IP:9000/" -i /home/bbpnrsoa/nrp/src/ExDFrontend/dist/config.json
cp /tum_nrp/nrp-docker/deploy/config/config.json /home/bbpnrsoa/nrp/src/nrpBackendProxy/config.json
source /home/bbpnrsoa/nrp/src/user-scripts/nrp_variables 2> /dev/null
/home/bbpnrsoa/nrp/src/user-scripts/add_new_database_storage_user -u nrpuser -p password -s > /dev/null 2>&1

# Starting NRP Frontend
echo "Starting NRP Frontend"
sudo /etc/init.d/supervisor stop
sudo /etc/init.d/supervisor start
sleep infinity
