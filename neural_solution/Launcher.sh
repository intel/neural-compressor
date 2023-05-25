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

#!/bin/bash
set +x

function main {

   init_params "$@"
   serve "$@"

}

check_port() {
  local port=$1
  if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 0 ] || [ "$port" -gt 65535 ]; then
    echo "Error: Invalid port number: $port"
    exit 1
  fi
}

# init params
function init_params {

   hostfile=None
   api_type=http # http, grpc, all
   restful_api_port=8000
   grpc_api_port=8001
   task_monitor_port=2222
   result_monitor_port=3333
   serve_log_dir=ns_workspace/serve_log
   workspace='./ns_workspace'
   upload_path=examples

   for var in "$@"
   do
      case $var in
         --hostfile=*)
         hostfile=$(echo $var |cut -f2 -d=)
         ;;
         --workspace=*)
         workspace=$(echo $var |cut -f2 -d=)
         ;;
         --restful_api_port=*)
         restful_api_port=$(echo $var |cut -f2 -d=)
         check_port $restful_api_port
         ;;
         --grpc_api_port=*)
         grpc_api_port=$(echo $var |cut -f2 -d=)
         check_port $grpc_api_port
         ;;
         --api_type=*)
         api_type=$(echo $var |cut -f2 -d=)
         ;;
         --task_monitor_port=*)
         task_monitor_port=$(echo $var |cut -f2 -d=)
         check_port $task_monitor_port
         ;;
         --result_monitor_port=*)
         check_port $result_monitor_port
         result_monitor_port=$(echo $var |cut -f2 -d=)
         ;;
         --conda_env=*)
         conda_env=$(echo $var |cut -f2 -d=)
         ;;
         --upload_path=*)
         upload_path=$(echo $var |cut -f2 -d=)
         ;;
         help|start|stop)
         ;;
         *)
            echo "Error: No such parameter: ${var}"
            exit 1
         ;;
      esac
   done

   # intialize constant.py


}


# serve runner
function serve {
  for var in "$@"
  do
    case $var in
      start)
         # Check ports
         ports_flag=0
         for port in $restful_api_port $task_monitor_port $result_monitor_port;
         do
            # Check if the port is occupied
            if lsof -i ":$port"  > /dev/null 2>&1; then
               echo "Port $port is in use!"
               ports_flag+=1
            # else
            #    echo "Port $port is available."
            fi
         done
         if [ $ports_flag -gt 0 ]; then
            echo "Please replace the occupied port!"
            exit 1
         fi
         # Check completed

         # Check conda environment
         if [ -z "$conda_env" ];then
            CONDA_ENV=$(echo $CONDA_DEFAULT_ENV)
            if [ -z "$CONDA_ENV" ];then
               echo "No environment specified or conda environment activated !!!"
               exit 1
            else
               echo "No environment specified, use environment activated: ($CONDA_ENV) as the task runtime environment."
               conda_env_name=$CONDA_ENV
            fi
         else
            conda_env_name=$conda_env
         fi

         # search package path
         # Check completed
         serve_log_dir=$workspace"/serve_log"
         mkdir -p $serve_log_dir
         date_suffix=_$(date +%Y%m%d-%H%M%S)
         date_suffix=
         >$serve_log_dir/backend$date_suffix.log
         >$serve_log_dir/frontend$date_suffix.log
         export PYTHONDONTWRITEBYTECODE=1 && python -m neural_solution.backend.Runner \
         --hostfile ${hostfile} \
         --task_monitor_port $task_monitor_port \
         --result_monitor_port $result_monitor_port \
         --workspace $workspace\
         --conda_env_name $conda_env_name \
         --upload_path $upload_path \
          >> $serve_log_dir/backend$date_suffix.log  2>&1 &
         export PYTHONDONTWRITEBYTECODE=1 && python -m neural_solution.frontend.fastapi.main_server \
         --host "0.0.0.0"\
         --fastapi_port $restful_api_port\
         --task_monitor_port $task_monitor_port\
         --result_monitor_port $result_monitor_port\
         --workspace $workspace\
         &>>$serve_log_dir/frontend$date_suffix.log &
         export PYTHONDONTWRITEBYTECODE=1 && python -m neural_solution.frontend.gRPC.server \
         --grpc_api_port $restful_api_port\
         --task_monitor_port $task_monitor_port\
         --result_monitor_port $result_monitor_port\
         --workspace $workspace\
         &>>$serve_log_dir/frontend_grpc.log &
         ip_address=$(hostname -I | awk '{print $1}')

         # Check if the service is started
         # Set the maximum waiting time to 3 senconds
         timeout=3
         # Start time
         start_time=$(date +%s)
         while true; do
               # Check if the ports are in use
               if ! lsof -i ":$task_monitor_port,$result_monitor_port" > /dev/null 2>&1; then
                 # If the ports are not in use, wait for a second and check again
                 sleep 0.5
                 # Check if timed out
                 current_time=$(date +%s)
                 elapsed_time=$((current_time - start_time))
                 if [ $elapsed_time -ge $timeout ]; then
                     # If timed out, break the loop
                     echo "Timeout!"
                     break
                 fi

                 # Continue to wait for the ports to be in use
                 continue
             fi

             break
         done
         ports_flag=0
         fail_msg="Neural Solution START FAIL!"
         for port in $task_monitor_port $result_monitor_port;
         do
            if lsof -i ":$port" > /dev/null 2>&1; then
               ports_flag=$((ports_flag+1))
            fi
         done

         # Check if the serve port is occupied
         if lsof -i ":$restful_api_port" > /dev/null 2>&1; then
            ports_flag=$((ports_flag+1))
         else
            fail_msg="$fail_msg\nPlease check frontend serve log!"
         fi

         if [ $ports_flag -lt 2 ]; then\
            fail_msg="$fail_msg\nPlease check backend serve log!"
         fi

         if [ $ports_flag -lt 3 ]; then
            echo -e $fail_msg
            exit 1
         fi
         # Check completed

         echo "Neural Solution Service Started!"
         echo "Service log saving path is in \"$(cd $serve_log_dir; pwd)\""
         echo "To submit task at: $ip_address:$restful_api_port/task/submit/"
         echo "[For information] ns help"

      ;;
      stop)

         # kill the remaining processes
         lsof -i | grep mpirun|awk '{print $2}' | xargs kill -9 > /dev/null 2>&1
         lsof -i | grep python|awk '{print $2}' | xargs kill -9 > /dev/null 2>&1

         # Service End
         echo "Neural Solution STOPPED!"
      ;;
      help)
         echo
         echo " *** usage: ns {start|stop} ***"
         echo "     start      : start serve"
         echo "     stop       : stop serve"
         echo
         echo "  more start parameters: [usage: ns start {--parameter=value}] [e.g. --restful_api_port=8000]"
         echo '    --hostfile           : start backend serve host file which contains all available nodes'
         echo '    --restful_api_port         : start web serve with {restful_api_port}, defult 8000'
         echo '    --api_type           : start web serve with grpc/http, defult http'
         echo '    --task_monitor_port  : start serve for task monitor at {task_monitor_port}, defult 2222'
         echo '    --result_monitor_port: start serve for result monitor at {result_monitor_port}, defult 3333'
         echo '    --workspace          : neural solution workspace, defult "./"'
         echo '    --conda_env          : specify the running environment for the task'
         echo '    --upload_path        : specify the file path for the tasks'

      ;;
      --hostfile=*|--restful_api_port=*|--result_monitor_port=*|--task_monitor_port=*|--api_type=*|*serve_log_dir=*|\
      --upload_path=*|--conda_env=*|--workspace=*)
      ;;
      *)
         echo "Error: No such parameter: ${var}"
         exit 1
      ;;
    esac
  done

}

main "$@"
