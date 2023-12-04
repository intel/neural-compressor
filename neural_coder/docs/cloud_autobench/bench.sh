# Copyright (c) 2022 Intel Corporation
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

# Read the user input  

check_input()
{
if  [ $whether_launch_new -eq "1" ]
then
  if  [ $vendor -eq "1" ]
  then
    if [ -z $security_id_aws ]
    then
      echo "[ERROR] There is no security group ID, you must specify security ID in config file when creating a new instance"
    fi
    
    if [ -z $subnet_id_aws ]
    then
      echo "[ERROR] There is no subnet ID, you must specify subnet ID in config file when creating a new instance"
    fi
  elif [ $vendor -eq "2" ]
  then
    if [ -z $security_id_ali ]
    then
      echo "[ERROR] There is no security group ID, you must specify security ID in config file when creating a new instance"
    fi
    
    if [ -z $region_id_ali ]
    then
      echo "[ERROR] There is no region ID, you must specify region ID in config file when creating a new instance"
    fi
  else
    echo "[ERROR] There is no this vendor"
  fi
else
  if [ -z $instance_id ]
  then
    echo "[ERROR] There is no instance ID, you must specify instance ID in config file when using an existed instance"
  fi
fi

}

create_AWS_instance()
{
if [ $os -eq "1" ]
then
   if [ $arch -eq "1" ]
     then
     ami_ID="ami-02f3416038bdb17fb"
   else
     ami_ID="ami-0ff596d41505819fd"
   fi
elif [ $os -eq "2" ]
then
    if [ $arch -eq "1" ]
     then
     ami_ID="ami-02d1e544b84bf7502"
   else
     ami_ID="ami-03e57de632660544c"
   fi
elif [ $os -eq "3" ]
then
    if [ $arch -eq "1" ]
     then
     ami_ID="ami-092b43193629811af"
   else
     ami_ID="ami-0082f8c86a7132597"
   fi
elif [ $os -eq "4" ]
then
    if [ $arch -eq "1" ]
     then
     ami_ID="ami-0f7cb53c916a75006"
   else
     ami_ID="ami-075a486be6269029f"
   fi
else 
   echo "[ERROR] The operating system is invalid"
   exit 0
fi
   
echo "[INFO] Starting creating AMS instance ..."
   
instance_id=$(aws ec2 run-instances --image-id $ami_ID --count $count --instance-type $i_type  --key-name $key_name --security-group-ids $security_id_aws  --subnet-id  $subnet_id_aws --block-device-mappings 'DeviceName=/dev/sda1, Ebs={VolumeSize=30}' --query "Instances[0].InstanceId")
result=$?
if [ $result -ne '0' ]
then
  echo "[ERROR] Create AWS Instance failed"
  exit 0
else
   echo "[INFO] Create AWS instance success"
   echo "[INFO] Your Instance Id is $instance_id"
   echo "[INFO] Waiting for instance to initialize ..."
   echo "[INFO] 15s left ..."
   sleep 5s
   echo "[INFO] 10s left ..."
   sleep 5s
   echo "[INFO] 5s left ..."
   sleep 5s
fi
} 

create_Ali_Yun_instance()
{
if [ $os -eq "1" ]
then
   if [ $arch -eq "1" ]
     then
     ami_ID="ubuntu_20_04_x64_20G_alibase_20220524.vhd"
   else
     ami_ID="ubuntu_20_04_x64_20G_alibase_20220524.vhd"
   fi
elif [ $os -eq "2" ]
then
    if [ $arch -eq "1" ]
     then
     ami_ID="ubuntu_20_04_x64_20G_alibase_20220524.vhd"
   else
     ami_ID="ubuntu_20_04_x64_20G_alibase_20220524.vhd"
   fi
elif [ $os -eq "3" ]
then
    if [ $arch -eq "1" ]
     then
     ami_ID="ubuntu_20_04_x64_20G_alibase_20220524.vhd"
   else
     ami_ID="ubuntu_20_04_x64_20G_alibase_20220524.vhd"
   fi
elif [ $os -eq "4" ]
then
    if [ $arch -eq "1" ]
     then
     ami_ID="ubuntu_20_04_x64_20G_alibase_20220524.vhd"
   else
     ami_ID="ubuntu_20_04_x64_20G_alibase_20220524.vhd"
   fi
else 
   echo "[ERROR] The operating system is invalid"
   exit 0
fi

i_type="ecs.$i_type_family.$i_type_size"
   
echo "[INFO] Starting creating Ali Yun instance ..."
   
instance_id=$(aliyun ecs RunInstances --RegionId $region_id_ali --InstanceType $i_type --InstanceChargeType PostPaid --ImageId $ami_ID --KeyPairName $key_name --SecurityGroupId $security_id_ali --VSwitchId vsw-m5ethlhigvonp2kuyzhjw --InternetMaxBandwidthIn 1 --InternetMaxBandwidthOut 1 |grep "i-")
result=$?

instance_id="${instance_id:4:22}"
if [ $result -ne '0' ]
then
  echo "[ERROR] Create Ali Yun Instance failed"
  exit 0
else
   echo "[INFO] Create Ali Yun instance successfully"
   echo "[INFO] The Ali Yun instance id is: $instance_id"
   echo "[INFO] Waiting for instance to initialize ..."
   echo "[INFO] 35s left ..."
   sleep 5s
   echo "[INFO] 30s left ..."
   sleep 5s
   echo "[INFO] 25s left ..."
   sleep 5s
   echo "[INFO] 20s left ..."
   sleep 5s
   echo "[INFO] 15s left ..."
   sleep 5s
   echo "[INFO] 10s left ..."
   sleep 5s
   echo "[INFO] 5s left ..."
   sleep 5s
fi
} 

connect_AWS()
{
dns_name=$(aws ec2 describe-instances --instance-ids $instance_id --query "Reservations[0].Instances[0].PublicDnsName")
result=$?
if [ $result -ne '0' ]
then
  echo "[ERROR] Can not find this instance, please check"
  exit 0
fi

host_name=$dns_name

if [ $os -eq "1" ]
then
  host_name="ubuntu@$dns_name"
else
  host_name="ec2-user@$dns_name"
fi

key_name="$key_name.pem"
echo "[INFO] Your instance host name is: $host_name"
echo "[INFO] Connecting to AWS Instance ..."
ssh -i $key_name $host_name -o "StrictHostKeyChecking no" "uname -a ; exit"
result=$?
if [ $result -ne '0' ]
then
  echo "[ERROR] SSH connection failed"
  echo "[INFO] Start terminating the Instance"
  aws ec2 terminate-instances --instance-ids $instance_id
  result=$?
  if [ $result -ne '0' ]
  then
    echo "[ERROR] Instance termination failed"
  else 
    echo "[INFO] Instance termination success"
  fi
  exit 0
else 
  echo "[INFO] Connect to AWS Instance success"
fi

echo "[INFO] Start to transferring benchmark files"
scp -i $key_name -r ./code/ $host_name:/tmp
result=$?
if [ $result -ne '0' ]
then
  echo "[ERROR] SSH connection failed"
  exit 0
else 
  echo "[INFO] File transferring success"
fi

if  [ $whether_launch_new -eq "1" ]
then
  ssh -i $key_name $host_name "cd /tmp/code; chmod +x ./config.sh; ./config.sh; exit"
  echo "[INFO] Install dependencies finished"
else
  echo "[INFO] Configured environment"
fi

echo "[INFO] Start launching the task ..."

ssh -i $key_name $host_name "cd /tmp/code; chmod +x ./launch.sh; ./launch.sh; exit"

echo "[INFO] Benchmark Execution finished"
}

connect_Ali_Yun()
{
public_ip=$(aliyun ecs DescribeInstances --output cols=InstanceId,PublicIpAddress.IpAddress rows=Instances.Instance[] |grep $instance_id)
result=$?
if [ $result -ne '0' ]
then
  echo "[ERROR] Can not find this instance, please check"
  exit 0
fi

public_ip="${public_ip:25}"
length=${#public_ip}
public_ip="${public_ip:1:$length-2}"
host_name="root@$public_ip"
key_name="$key_name.pem"
echo "[INFO] Your instance host name is: $host_name"

echo "[INFO] Start to connecting Ali Yun instance"
ssh -i $key_name $host_name -o "StrictHostKeyChecking no" "uname -a ; exit"
result=$?
if [ $result -ne '0' ]
then
  echo "[ERROR] SSH connection failed"
  echo "[INFO] Start to delete instance $instance_id"
  sleep 60s
  aliyun ecs DeleteInstance --InstanceId $instance_id --Force true
  result=$?
  if [ $result -ne '0' ]
  then
    echo "[ERROR] Instance termination failed"
    exit 0
  else 
    echo "[INFO] Instance termination success"
  fi
  exit 0
else 
  echo "[INFO] Connect to Ali Yun Instance success"
fi

echo "[INFO] Start to transferring benchmark files"
scp -i $key_name -r ./code/ $host_name:/tmp
result=$?
if [ $result -ne '0' ]
then
  echo "[ERROR] SSH connection failed"
  exit 0
else 
  echo "[INFO] File transferring success"
fi

if  [ $whether_launch_new -eq "1" ]
then
  ssh -i $key_name $host_name "cd /tmp/code; chmod +x ./config.sh; ./config.sh; exit"
  echo "[INFO] Install dependencies finished"
else
  echo "[INFO] Configured environment"
fi

echo "[INFO] Start launching the task ..."

ssh -i $key_name $host_name "cd /tmp/code; chmod +x ./launch.sh; ./launch.sh; exit"

echo "[INFO] Benchmark Execution finished"
}

close_AWS()
{

if [ $whether_retain -eq "1" ]
then
  echo "[INFO] Start stopping the Instance"

  aws ec2 stop-instances --instance-ids $instance_id
  result=$?
  if [ $result -ne '0' ]
  then
    echo "[ERROR] Instance stop failed"
    exit 0
  else 
    echo "[INFO] Instance stop success"
    echo "[INFO] The instance id is $instance_id, Please record this $instance_id for next use"
  fi
else 
  echo "[INFO] Start terminating the Instance"

  aws ec2 terminate-instances --instance-ids $instance_id
  result=$?
  if [ $result -ne '0' ]
  then
    echo "[ERROR] Instance termination failed"
    exit 0
  else 
    echo "[INFO] Instance termination success"
  fi
fi
}

close_Ali_Yun()
{

if [ $whether_retain -eq "1" ]
then
  echo "[INFO] Start stopping the Instance"

  aliyun ecs StopInstance --InstanceId $instance_id
  result=$?
  if [ $result -ne '0' ]
  then
    echo "[ERROR] Instance stop failed"
    exit 0
  else 
    echo "[INFO] Instance stop success"
    echo "[INFO] The instance id is $instance_id, Please record this $instance_id for next use"
  fi
elif [ $whether_retain -eq "2" ]
then
  echo "[INFO] Start terminating the Instance"

  aliyun ecs DeleteInstance --InstanceId $instance_id --Force true
  result=$?
  if [ $result -ne '0' ]
  then
    echo "[ERROR] Instance termination failed"
    exit 0
  else 
    echo "[INFO] Instance termination success"
  fi
fi
}


main()
{
vendor=$(sed '/^cloud_vendor=/!d; s/.*=//' config.conf)
os=$(sed '/^OS=/!d; s/.*=//' config.conf)
arch=$(sed '/^arch=/!d; s/.*=//' config.conf)
count=$(sed '/^count=/!d; s/.*=//' config.conf)
i_type_family=$(sed '/^i_type_family=/!d; s/.*=//' config.conf)
i_type_size=$(sed '/^i_type_size=/!d; s/.*=//' config.conf)
key_name=$(sed '/^key_name=/!d; s/.*=//' config.conf)
instance_id=$(sed '/^instance_id=/!d; s/.*=//' config.conf)
security_id_aws=$(sed '/^security_id_aws=/!d; s/.*=//' config.conf)
subnet_id_aws=$(sed '/^subnet_id_aws=/!d; s/.*=//' config.conf)
security_id_ali=$(sed '/^security_id_ali=/!d; s/.*=//' config.conf)
region_id_ali=$(sed '/^region_id_ali=/!d; s/.*=//' config.conf)

whether_retain=$(sed '/^whether_retain=/!d; s/.*=//' config.conf)
whether_launch_new=$(sed '/^whether_launch_new=/!d; s/.*=//' config.conf)

i_type="$i_type_family.$i_type_size"

check_input

if [ ! -f "$key_name.pem" ]; then
  echo "[ERROR] Can not find the key pair file $key_name.pem, please put the $key_name.pem file in this folder"
  exit 0
else
  chmod 400 ./"$key_name.pem"
fi

if [ ! -f "./code/benchmark.py" ]; then
  echo "[ERROR] Can not find the benchmark file, please put the benchmark file in code folder"
  exit 0
fi


if [ $whether_launch_new -eq "1" ]
then
  echo "[INFO] Your instance info:"
  echo "[INFO] Instance key name: $key_name"
  echo "[INFO] Instance count: $count"
  echo "[INFO] Instance_type: $i_type"
else
  echo "[INFO] The existed instance you choose: $instance_id"
fi

if [ $whether_launch_new -eq "1" ]
then
  if [ $vendor -eq "1" ]
  then
    create_AWS_instance
  elif [ $vendor -eq "2" ]
  then
    create_Ali_Yun_instance
  else
    echo "Tencent Cloud"
  fi
else
  if [ $vendor -eq "1" ]
  then
    aws ec2 start-instances --instance-ids $instance_id
    echo "[INFO] Waiting for instance to Start ..."
    echo "[INFO] 15s left ..."
    sleep 5s
    echo "[INFO] 10s left ..."
    sleep 5s
    echo "[INFO] 5s left ..."
    sleep 5s
  elif [ $vendor -eq "2" ]
  then
    aliyun ecs StartInstance --InstanceId $instance_id
    echo "[INFO] Waiting for instance to Start ..."
    echo "[INFO] 45s left ..."
    sleep 15s
    echo "[INFO] 30s left ..."
    sleep 15s
    echo "[INFO] 15s left ..."
    sleep 15s
  else
    echo "Tencent Cloud"
  fi
fi

if [ $vendor -eq "1" ]
then
  connect_AWS
elif [ $vendor -eq "2" ]
then
  connect_Ali_Yun
else 
  echo "Tencent Cloud"
fi

if [ $vendor -eq "1" ]
then
  close_AWS
elif [ $vendor -eq "2" ]
then
  close_Ali_Yun
else
  echo "Tencent Cloud"
fi

exit 0

}

main




