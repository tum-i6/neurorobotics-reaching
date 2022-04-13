import subprocess

def get_host_ip_address():
    """
    Function that determines the IPaddress of docker-container.
    
    return
      - ip-address
    """
    systemcall = subprocess.Popen("hostname -I", shell=True, stdout=subprocess.PIPE)
    ip_address = systemcall.stdout.read().strip().decode('ascii')
    return ip_address

def get_grpc_server_port(port=50051):
    """
    Function that returns the grpc server port to bind on.
    
    return
      - server_port: "IP:port"
    """
    server_port = str(get_host_ip_address()) + ':' + str(port)
    return server_port
    
def ip_address_adapter(ip_address):
    ips = ip_address.split(".")
    ips[3] = str(int(ips[3])+1) # this changes to host_id by 1
    return ".".join(ips)
    
