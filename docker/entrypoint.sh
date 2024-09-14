#!/bin/bash  
  
# Source the setup script  
source /IRON/setup.sh  
  
# Execute the command passed to the container  
exec "$@" 
