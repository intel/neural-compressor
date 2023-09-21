# Cloud Auto-bench
This is a user guide for the automated bash script for creating a cloud instance, configuring the environment, running the benchmark code, and terminating the instance. The script supports AWS and Ali Yun for now and will support more cloud vendors in the future.

## Prerequisite
## AWS
#### Install AWS CLI
Install the latest AWS CLI according to https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html.
Example: install AWS CLI on Linux x86 (64-bit) by the following commands.
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install -i ~/.Awscli2 -b ~/.Awscli2/bin
export PATH=/.Awscli2/bin:$PATH
```
Confirm the installation with the following command.
```
export PATH=/.Awscli2/bin:$PATH
```

#### Configure IAM role
Configure IAM role to access AWS according to https://us-east-2.console.aws.amazon.com/iamv2/home#/users.

#### Configure AWS clients
Configure AWS clients via "aws configure" which fetch from step 2.
```
aws --configure
AWS Access Key ID [None]: x
AWS Secret Access Key [None]: x
Default region name [None]: us-east-2
Default output format [None]: text
```

#### Create a key pair
Create a key pair x.pem via https://us-east-2.console.aws.amazon.com/ec2/v2/home?region=us-east-2#KeyPairs:, add x.pem to the current folder, and modify its permission by 
```
chmod 400 x.pem
```

## Ali Yun
#### Install Alibaba Cloud CLI
Download the installation package for Linux from https://www.alibabacloud.com/help/en/alibaba-cloud-cli/latest/linux

Decompress the downloaded file to obtain the executable file named aliyun
```
tar xzvf (aliyun-cli-linux-latest-amd64.tgz) 
```
In the "()" is the file name of the installation file you have downloaded

Set environment variables
add this line into the ~/.bash_profile file
```
export PATH=/home/your_directory/.Awscli2/bin:$PATH  
```

Your directory is the directory of bin in the ALi Yun installation directory)
source environment variables

#### Configure Ali Yun clients
run 
```
aliyun configure
```

Input the key ID, key secret, region ID and default language of your Ali Yun account
```
Access Key Id [************************]:
Access Key Secret [******************************]:
Default Region Id [**-****]:
Default Output Format [json]: json (Only support json)
Default Language [zh|en] en
```

### Create a key pair of Ali Yun according to page https://ecs.console.aliyun.com/#/keyPair/
add x.pem to the current folder, and modify its permission by 
```
chmod 400 x.pem
```

## Launch the cloud benchmark script
### Modify the permission of bench.sh
```
chmod 755 bench.sh
```

### Input your task config information in the config.conf file
You need to input the information of your task in the config.conf file
You can choose the cloud vendor(AWS or Ali Yun), Instance number, type and other information with the comments as a reference

### Launch the script
```
./bench.sh
```

### Output
Example Cloud auto-bench report: ```superbench_report_aws_icx.pdf```
