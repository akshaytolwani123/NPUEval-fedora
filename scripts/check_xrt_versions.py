# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import sys
import os

def check_xrt_versions():
   '''Helper function to make sure XRT versions in the container
   and host system are the same to avoid device driver issues.
   '''
   host_file = "/host/version.json"
   container_file = "/opt/xilinx/xrt/share/amdxdna/version.json"
   
   if not os.path.exists(host_file):
       print(f"WARNING: Host version file not found at {host_file}")
       sys.exit(1)
   
   if not os.path.exists(container_file):
       print(f"WARNING: Container version file not found at {container_file}")
       sys.exit(1)
   
   try:
       with open(host_file, 'r') as f:
           host_data = json.load(f)
       
       with open(container_file, 'r') as f:
           container_data = json.load(f)
       
       # Comparing XRT and AMDXDNA commits -- should match on host and container
       host_xrt_hash = host_data.get('XRT_LAST_COMMIT_HASH', 'unknown')
       container_xrt_hash = container_data.get('XRT_LAST_COMMIT_HASH', 'unknown')
       
       host_last_hash = host_data.get('LAST_COMMIT_HASH', 'unknown')
       container_last_hash = container_data.get('LAST_COMMIT_HASH', 'unknown')
       
       # Exit 1 if there's a mismatch, minor commits aren't necessarily a dealbreaker
       # but can cause issues and throw DRM_IOCTL_AMDXDNA_EXEC_CMD style errors
       if host_xrt_hash != container_xrt_hash or host_last_hash != container_last_hash:
           print("WARNING: XRT commit hash mismatch detected")
           print(f"Host XRT_LAST_COMMIT_HASH: {host_xrt_hash}")
           print(f"Container XRT_LAST_COMMIT_HASH: {container_xrt_hash}")
           print(f"Host LAST_COMMIT_HASH: {host_last_hash}")
           print(f"Container LAST_COMMIT_HASH: {container_last_hash}")
           print("Exiting due to version mismatch")
           sys.exit(1)
       
   except Exception as e:
       print(f"Error checking versions: {e}")
       sys.exit(1)

if __name__ == "__main__":
   check_xrt_versions()
