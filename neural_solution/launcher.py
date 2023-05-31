# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import subprocess
import os
import argparse
import socket
import psutil
import time
from datetime import datetime

def check_port(port):
    if not str(port).isdigit() or int(port) < 0 or int(port) > 65535:
        print(f"Error: Invalid port number: {port}")
        sys.exit(1)

def get_local_service_ip(port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]

def stop_service():
    # Get all running processes
    for proc in psutil.process_iter():
        try:
            # Get the process details
            pinfo = proc.as_dict(attrs=['pid', 'name', 'cmdline'])
            # Check if the process is the target process
            if "neural_solution.backend.runner" in pinfo['cmdline']:
                # Terminate the process using Process.kill() method
                process = psutil.Process(pinfo['pid'])
                process.kill()
            elif "neural_solution.frontend.fastapi.main_server" in pinfo['cmdline']:
                process = psutil.Process(pinfo['pid'])
                process.kill()
            elif "neural_solution.frontend.gRPC.server" in pinfo['cmdline']:
                process = psutil.Process(pinfo['pid'])
                process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def check_port_free(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        sock.bind(("0.0.0.0", port))
        result = True
    except:
        # Port is in use
        pass 
    sock.close()
    return result

def serve(args):

    if args.action == "start":
        # Check ports
        ports_flag = 0
        for port in [args.restful_api_port, args.task_monitor_port, args.result_monitor_port]:
            # Check if the port is occupied
            if not check_port_free(port):
                print(f"Port {port} is in use!")
                ports_flag += 1
        if ports_flag > 0:
            print("Please replace the occupied port!")
            sys.exit(1)
        # Check completed

        # Check conda environment
        if not args.conda_env:
            conda_env = os.environ.get("CONDA_DEFAULT_ENV")
            if not conda_env:
                print("No environment specified or conda environment activated !!!")
                sys.exit(1)
            else:
                print(f"No environment specified, use environment activated:" + \
                    f" ({conda_env}) as the task runtime environment.")
                conda_env_name = conda_env
        else:
            conda_env_name = args.conda_env
        # Check completed

        serve_log_dir = f"{args.workspace}/serve_log"
        os.makedirs(serve_log_dir, exist_ok=True)
        date_time = datetime.now()
        date_suffix = "_" + date_time.strftime("%Y%m%d-%H%M%S")
        date_suffix = ""
        with open(f"{serve_log_dir}/backend{date_suffix}.log", "w") as f:
            subprocess.Popen([
                "python", "-m", "neural_solution.backend.runner",
                "--hostfile", str(args.hostfile),
                "--task_monitor_port", str(args.task_monitor_port),
                "--result_monitor_port", str(args.result_monitor_port),
                "--workspace", str(args.workspace),
                "--conda_env_name", str(conda_env_name),
                "--upload_path", str(args.upload_path)
            ], stdout=f, stderr=subprocess.STDOUT)
        with open(f"{serve_log_dir}/frontend{date_suffix}.log", "w") as f:
            subprocess.Popen([
                "python", "-m", "neural_solution.frontend.fastapi.main_server",
                "--host", "0.0.0.0",
                "--fastapi_port", str(args.restful_api_port),
                "--task_monitor_port", str(args.task_monitor_port),
                "--result_monitor_port", str(args.result_monitor_port),
                "--workspace", str(args.workspace)
            ], stdout=f, stderr=subprocess.STDOUT)
        with open(f"{serve_log_dir}/frontend_grpc.log", "w") as f:
            subprocess.Popen([
                "python", "-m", "neural_solution.frontend.gRPC.server",
                "--grpc_api_port", str(args.grpc_api_port),
                "--task_monitor_port", str(args.task_monitor_port),
                "--result_monitor_port", str(args.result_monitor_port),
                "--workspace", str(args.workspace)
            ], stdout=f, stderr=subprocess.STDOUT)
        ip_address = get_local_service_ip(80)

        # Check if the service is started
        # Set the maximum waiting time to 3 senconds
        timeout = 3
        # Start time
        start_time = time.time()
        while True:
            # Check if the ports are in use
            if check_port_free(args.task_monitor_port) and check_port_free(args.result_monitor_port) \
                and check_port_free(args.restful_api_port):
                # If the ports are not in use, wait for a second and check again
                time.sleep(0.5)
                # Check if timed out
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= timeout:
                    # If timed out, break the loop
                    print("Timeout!")
                    break

                # Continue to wait for the ports to be in use
                continue

            break
        ports_flag = 0
        fail_msg = "Neural Solution START FAIL!"
        for port in [args.task_monitor_port, args.result_monitor_port]:
            if not check_port_free(port):
                ports_flag += 1

        # Check if the serve port is occupied
        if check_port_free(args.restful_api_port):
            ports_flag += 1
        else:
            fail_msg = f"{fail_msg}\nPlease check frontend serve log!"

        if ports_flag < 2:
            fail_msg = f"{fail_msg}\nPlease check backend serve log!"

        if ports_flag < 3:
            print(fail_msg)
            sys.exit(1)
        # Check completed

        print("Neural Solution Service Started!")
        print(f"Service log saving path is in \"{os.path.abspath(serve_log_dir)}\"")
        print(f"To submit task at: {ip_address}:{args.restful_api_port}/task/submit/")
        print("[For information] neural_solution help")

    elif args.action == "stop":
        # kill the remaining processes
        stop_service()
        # Service End
        print("Neural Solution Service Stopped!")

def main():
    parser = argparse.ArgumentParser(description="Neural Solution")
    parser.add_argument('action', choices=['start', 'stop', 'help'], help='Action to perform')
    parser.add_argument("--hostfile", default=None,
                        help="start backend serve host file which contains all available nodes")
    parser.add_argument("--restful_api_port", type=int, default=8000,
                        help="start restful serve with {restful_api_port}, default 8000")
    parser.add_argument("--grpc_api_port", type=int, default=8001,
                        help="start gRPC with {restful_api_port}, default 8000")
    parser.add_argument("--result_monitor_port", type=int, default=3333,
                        help="start serve for result monitor at {result_monitor_port}, default 3333")
    parser.add_argument("--task_monitor_port", type=int, default=2222,
                        help="start serve for task monitor at {task_monitor_port}, default 2222")
    parser.add_argument("--api_type", default="all",
                        help="start web serve with all/grpc/restful, default all")
    parser.add_argument("--workspace", default="./ns_workspace", 
                        help="neural solution workspace, default \"./ns_workspace\"")
    parser.add_argument("--conda_env", default=None, help="specify the running environment for the task")
    parser.add_argument("--upload_path", default="examples", help="specify the file path for the tasks")
    args = parser.parse_args()

    check_port(args.restful_api_port)
    check_port(args.grpc_api_port)
    check_port(args.result_monitor_port)
    check_port(args.task_monitor_port)

    if args.action == 'start' or args.action == 'stop' :
        serve(args=args)
    elif args.action == 'help':
        print("\n *** usage: neural_solution {start|stop} ***")
        print("     start      : start serve")
        print("     stop       : stop serve\n")
        print("  more start parameters: [usage: neural_solution start {--parameter=value}] [e.g. --restful_api_port=8000]")
        print('    --hostfile           : start backend serve host file which contains all available nodes')
        print('    --restful_api_port   : start web serve with {restful_api_port}, default 8000')
        print('    --api_type           : start web serve with all/grpc/restful, default all')
        print('    --task_monitor_port  : start serve for task monitor at {task_monitor_port}, default 2222')
        print('    --workspace          : neural solution workspace, default "./ns_workspace"')
        print('    --conda_env          : specify the running environment for the task"')
        print('    --upload_path        : specify the file path for the tasks')
        
if __name__ == '__main__':
    main()
