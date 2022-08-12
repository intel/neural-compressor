# Cloud Auto-bench
This is a user guide of the automated bash script for creating a cloud instance, configuring the environment, running the benchmark code, and terminating the instance.

## AWS
### Prerequisite
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

#### Configure .ssh/config
Configure .ssh/config for accessing the cloud instances using the public IP address and add the below config file to .ssh/config.
```
Host * !*.intel.com !10.*.*.*
ProxyCommand nc --proxy-type socks4 --proxy proxy-dmz.intel.com:1080 %h %p
```
```
touch .ssh/config && chmod 700 .ssh/config
```

#### Create a key pair
Create a key pair x.pem via https://us-east-2.console.aws.amazon.com/ec2/v2/home?region=us-east-2#KeyPairs:, add x.pem to the current folder, and modify its permission by 
```
chmod 400 x.pem
```

### Modify the permission of bench.sh and run
```
chmod 755 bench.sh
./bench.sh
```

### Output
Example Cloud auto-bench report: ```superbench_report_aws_icx.pdf```
