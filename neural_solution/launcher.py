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

def check_port(port):
    if not str(port).isdigit() or int(port) < 0 or int(port) > 65535:
        print(f"Error: Invalid port number: {port}")
        sys.exit(1)

def init_params(args):
    hostfile = None
    api_type = "http"
    restful_api_port = 8000
    grpc_api_port = 8001
    task_monitor_port = 2222
    result_monitor_port = 3333
    serve_log_dir = "ns_workspace/serve_log"
    workspace = "./ns_workspace"
    upload_path = "examples"

    for arg in vars(args):
        if arg == "hostfile":
            hostfile = getattr(args, arg)
        elif arg == "workspace":
            workspace = getattr(args, arg)
        elif arg == "restful_api_port":
            restful_api_port = getattr(args, arg)
            check_port(restful_api_port)
        elif arg == "grpc_api_port":
            grpc_api_port = getattr(args, arg)
            check_port(grpc_api_port)
        elif arg == "api_type":
            api_type = getattr(args, arg)
        elif arg == "task_monitor_port":
            task_monitor_port = getattr(args, arg)
            check_port(task_monitor_port)
        elif arg == "result_monitor_port":
            result_monitor_port = getattr(args, arg)
            check_port(result_monitor_port)
        elif arg == "conda_env":
            conda_env = getattr(args, arg)
        elif arg == "upload_path":
            upload_path = getattr(args, arg)
        elif arg in ["action"]:
            pass
        else:
            print(f"Error: No such parameter: {arg}")
            sys.exit(1)

    return hostfile, api_type, restful_api_port, grpc_api_port, task_monitor_port, result_monitor_port, \
        serve_log_dir, workspace, upload_path,

def serve(args):
    hostfile, api_type, restful_api_port, grpc_api_port, task_monitor_port, result_monitor_port, \
    serve_log_dir, workspace, upload_path = init_params(args)

    if args.action == "start":
        # Check ports
        ports_flag = 0
        for port in [restful_api_port, task_monitor_port, result_monitor_port]:
            # Check if the port is occupied
            if subprocess.run(["lsof", "-i", f":{port}"], stdout=subprocess.PIPE).stdout.decode():
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
                print(f"No environment specified, use environment activated: ({conda_env}) as the task runtime environment.")
                conda_env_name = conda_env
        else:
            conda_env_name = args.conda_env

        # search package path
        # Check completed
        serve_log_dir = f"{workspace}/serve_log"
        os.makedirs(serve_log_dir, exist_ok=True)
        date_suffix = "_"+subprocess.run(["date", "+%Y%m%d-%H%M%S"], stdout=subprocess.PIPE).stdout.decode().strip()
        date_suffix = ""
        with open(f"{serve_log_dir}/backend{date_suffix}.log", "w") as f:
            subprocess.Popen([
                "python", "-m", "neural_solution.backend.runner",
                "--hostfile", str(hostfile),
                "--task_monitor_port", str(task_monitor_port),
                "--result_monitor_port", str(result_monitor_port),
                "--workspace", str(workspace),
                "--conda_env_name", str(conda_env_name),
                "--upload_path", str(upload_path)
            ], stdout=f, stderr=subprocess.STDOUT)
        with open(f"{serve_log_dir}/frontend{date_suffix}.log", "w") as f:
            subprocess.Popen([
                "python", "-m", "neural_solution.frontend.fastapi.main_server",
                "--host", "0.0.0.0",
                "--fastapi_port", str(restful_api_port),
                "--task_monitor_port", str(task_monitor_port),
                "--result_monitor_port", str(result_monitor_port),
                "--workspace", str(workspace)
            ], stdout=f, stderr=subprocess.STDOUT)
        with open(f"{serve_log_dir}/frontend_grpc.log", "w") as f:
            subprocess.Popen([
                "python", "-m", "neural_solution.frontend.gRPC.server",
                "--grpc_api_port", str(grpc_api_port),
                "--task_monitor_port", str(task_monitor_port),
                "--result_monitor_port", str(result_monitor_port),
                "--workspace", str(workspace)
            ], stdout=f, stderr=subprocess.STDOUT)
        ip_address = subprocess.run(["hostname", "-I"], stdout=subprocess.PIPE).stdout.decode().strip().split()[0]

        # Check if the service is started
        # Set the maximum waiting time to 3 senconds
        timeout = 3
        # Start time
        start_time = int(subprocess.run(["date", "+%s"], stdout=subprocess.PIPE).stdout.decode().strip())
        while True:
            # Check if the ports are in use
            if not subprocess.run(["lsof", "-i", f":{task_monitor_port},{result_monitor_port}"], stdout=subprocess.PIPE).stdout.decode():
                # If the ports are not in use, wait for a second and check again
                subprocess.run(["sleep", "0.5"])
                # Check if timed out
                current_time = int(subprocess.run(["date", "+%s"], stdout=subprocess.PIPE).stdout.decode().strip())
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
        for port in [task_monitor_port, result_monitor_port]:
            if subprocess.run(["lsof", "-i", f":{port}"], stdout=subprocess.PIPE).stdout.decode():
                ports_flag += 1

        # Check if the serve port is occupied
        if subprocess.run(["lsof", "-i", f":{restful_api_port}"], stdout=subprocess.PIPE).stdout.decode():
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
        print(f"To submit task at: {ip_address}:{restful_api_port}/task/submit/")
        print("[For information] neural_solution help")

    elif args.action == "stop":
        # kill the remaining processes
        subprocess.run("lsof -i | grep mpirun|awk '{print $2}' | xargs kill -9 > /dev/null 2>&1", shell=True)
        subprocess.run("lsof -i | grep python|awk '{print $2}' | xargs kill -9 > /dev/null 2>&1", shell=True)
        # Service End
        print("Neural Solution Service Stopped!")

def main():
    parser = argparse.ArgumentParser(description="Neural Solution")
    parser.add_argument('action', choices=['start', 'stop', 'help'], help='Action to perform')
    parser.add_argument("--hostfile", default=None, help="start backend serve host file which contains all available nodes")
    parser.add_argument("--restful_api_port", type=int, default=8000, help="start web serve with {restful_api_port}, default 8000")
    parser.add_argument("--result_monitor_port", type=int, default=3333, help="start serve for result monitor at {result_monitor_port}, default 3333")
    parser.add_argument("--task_monitor_port", type=int, default=2222, help="start serve for task monitor at {task_monitor_port}, default 2222")
    parser.add_argument("--api_type", default="http", help="start web serve with grpc/http, default http")
    parser.add_argument("--workspace", default="./ns_workspace", help="neural solution workspace, default \"./ns_workspace\"")
    parser.add_argument("--conda_env", default=None, help="specify the running environment for the task")
    parser.add_argument("--upload_path", default="examples", help="specify the file path for the tasks")
    args = parser.parse_args()


    if args.action == 'start' or args.action == 'stop' :
        serve(args=args)
    elif args.action == 'help':
        print("\n *** usage: neural_solution {start|stop} ***")
        print("     start      : start serve")
        print("     stop       : stop serve\n")
        print("  more start parameters: [usage: neural_solution start {--parameter=value}] [e.g. --restful_api_port=8000]")
        print('    --hostfile           : start backend serve host file which contains all available nodes')
        print('    --restful_api_port   : start web serve with {restful_api_port}, default 8000')
        print('    --api_type           : start web serve with grpc/http, default http')
        print('    --task_monitor_port  : start serve for task monitor at {task_monitor_port}, default 2222')
        print('    --workspace          : start web serve with grpc/http, default http')
        print('    --conda_env          : start web serve with grpc/http, default http')
        print('    --upload_path        : start web serve with grpc/http, default http')


if __name__ == '__main__':
    main()
