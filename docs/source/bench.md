Intel® Neural Compressor Bench
=======

Intel® Neural Compressor Bench is a web application for easier use of Intel® Neural Compressor.

## Table of Contents
1. [Introduction](#introduction)
2. [Home Screen](#home-screen)
3. [Create New Project](#create-new-project)

    3.1. [Optimization Tab](#optimization-tab)

    3.2. [Benchmark Tab](#benchmark-tab)

    3.3. [Profiling Tab](#profiling-tab)

    3.4. [Diagnosis Tab](#diagnosis-tab)

    3.5. [Dataset Tab](#dataset-tab)

    3.6. [Project Information](#project-information-tab)

4. [NAS demo](#nas-demo)
5. [System Information](#system-information)
6. [Security](#security)

# Introduction
## Install Intel® Neural Compressor with Bench
To install Install Intel® Neural Compressor with GUI install full version of Intel® Neural Compressor in one of following ways:
### Option 1: Install from Binary

```Shell
# install stable full version from pip (including GUI)
pip install neural-compressor-full

# install nightly full version from pip (including GUI)
pip install -i https://test.pypi.org/simple/ neural-compressor-full

# install stable full version from from conda (including GUI)
conda install neural-compressor-full -c conda-forge -c intel  
```

### Option 2: Install from Source

```Shell
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -r requirements.txt
# build with full functionality (including GUI)
python setup.py --full install
```

## Start the Intel® Neural Compressor Bench

To start the Intel® Neural Compressor Bench server execute `inc_bench` command:

```shell
inc_bench
```
The server generates a self-signed TLS certificate and prints instruction how to access the Web UI.

```text
Intel(r) Neural Compressor Bench Server started.

Open address https://10.11.12.13:5000/?token=338174d13706855fc6924cec7b3a8ae8
```

Server generated certificate is not trusted by your web browser, you will need to accept usage of such certificate.


You might also use additional parameters and settings:
* Intel® Neural Compressor Bench listens on port 5000.
Make sure that port 5000 is accessible to your browser (you might need to open it in your firewall),
or specify different port that is already opened, for example 8080:
    ```shell
    inc_bench -p 8080
    ```

* When using official TF>=2.6.0, set environment variable `TF_ENABLE_ONEDNN_OPTS=1` for INT8 tuning:
    ```shell
    TF_ENABLE_ONEDNN_OPTS=1 inc_bench
    ```

* To start the Intel® Neural Compressor Bench server with your own TLS certificate add `--cert` and `--key` parameters:

    ```shell
    inc_bench --cert path_to_cert.crt --key path_to_private_key.key
    ```

* To start the Intel® Neural Compressor Bench server without TLS encryption use `--allow-insecure-connections` parameter:

    ```shell
    inc_bench --allow-insecure-connections
    ```

    This enables access to the server from any machine in your local network (or the whole Internet if your server is exposed to it).

    You are forfeiting security, confidentiality and integrity of all client-server communication. Your server is exposed to external threats.

## Home Screen
This view shows introduction to Intel® Neural Compressor Bench and a button for creating new project. After clicking this button, pop-up with project wizard will be shown.

![Home](imgs/bench/home.png "Home")

# Create New Project
To create a new project, in first step you need to choose its name.

![Project1](imgs/bench/project1.png "Project1")

In second step there are 2 possible options to choose from: 
* *predefined model* - you choose model from predefined examples list, you don't need to set any additional parameters,
* *custom model* - in this scenario you can set more parameters and customize your model. 

![Project2](imgs/bench/project2.png "Project2")

## Predefined Model
First you need to choose domain for the model (image recognition or object detection). For each domain there are few available models to choose from. When you click *Finish* the chosen model will be downloaded.

![Project3-predefined](imgs/bench/project3-predefined.png "Project3-predefined")

## Custom Model

First you have to choose the model path. When it is chosen, in most cases all other fields will be completed automatically. You can edit its input and output nodes, see the model graph (if it is available for this model) and set shape for synthetic dataset. If model domain was not detected, you need to choose it from the list. Model domain is used to set some default parameters for the model.

![Project3-custom](imgs/bench/project3-custom.png "Project3-custom")


## Display Model Graph
For several model types there will be a button available ![Show graph](imgs/bench/show_graph_button.png "Show graph") in the project wizard. It is also possible to see the graph in Diagnosis tab. The graph by default is collapsed, but when you click on plus icon, sections will be unfolded.


![Resnet model graph](imgs/bench/graph-resnet.jpg "Resnet model graph").

# Project List
On the left hand side there is a panel with list of created projects. When you click on the project name, you can see its details. "Create new project" button navigates to new project wizard pop-up described in previous section.

![Menu](imgs/bench/menu.png "Menu")

## Remove Project

If you want to remove project, you have to click the trash icon next to project name (it is visible when the cursor is on the project name).

![Remove1](imgs/bench/remove1.png "Remove1")

Then you will be prompted to confirm your choice by typing the project name. Project removal is not reversible.

![Remove2](imgs/bench/remove2.png "Remove2")

# Develop Project

## Optimization Tab
### Optimization Table
In Optimizations tab you can see list of optimizations in the project. Currently UI supports three optimization precisions and two types of optimization.
![Optimizations-table](imgs/bench/optimizations-table.png "Optimizations-table")

### Optimization Wizard
To add new optimization, click "Add new optimization" button at the bottom of the table and follow the steps.
![Optimizations-wizard](imgs/bench/optimizations-wizard.png "Optimizations-wizard")

### Editing Optimization Entries
There is a possibility to modify some optimization parameters even after exit from Wizard.
If optimization has not been run yet, the pencil icon on the right hand side should be in light blue color. That indicates that it can be modified. After click on that pencil icon you can select different precision or dataset.

For Quantization you can also modify Tuning details before optimizing model.
![Optimization-tuning-details](imgs/bench/optimization-tuning-details.png "Optimization-tuning-details")

### Optimization Details
To perform optimization click "Run" button. Once process is finished you can click on row with specific optimization to display details about optimization parameters and optimized model. When you click on blue arrow icon in model path line, you can download optimized model.
![Optimization-details](imgs/bench/optimization-details.png "Optimization-details")

### Pruning
To add pruning, click "Add new optimization" and in the *Precision* step choose *fp32* and then check the *pruning* option. Choose the dataset and click *Finish*.
![Pruning-wizard](imgs/bench/pruning-wizard.png "Pruning-wizard")
Pruning parameters are visible after in *Optimization details* section described before.
![Pruning-details](imgs/bench/pruning-details.png "Pruning-details")
Click *Run* button in the optimization table to execute the pruning. When the pruning is finished, path to the optimized model can be found in the details section.

## Benchmark Tab
### Benchmark Table
For each optimization and input model you can add benchmark. Benchmark have 2 modes: accuracy and performance. In benchmark tab you can see all your benchmarks. When you check checkboxes in the last column you can choose benchmark you want to compare in the chart (visible after clicking "Compare selected").

![Benchmarks-table](imgs/bench/benchmarks-table.png "Benchmarks-table")

### Benchmark Wizard
To add new benchmark, click "Add new benchmark" button at the bottom of the table and follow the steps.

![Benchmarks-wizard](imgs/bench/benchmarks-wizard.png "Benchmarks-wizard")


### Editing Benchmark Entries
As for optimizations you can also modify benchmark parameters. You can modify benchmark mode, dataset and benchmark parameters like batch size, number of instances and number of cores per instance.
![Benchmark-edit-wizard](imgs/bench/benchmark-edit-wizard.png "Benchmark-edit-wizard")

### Benchmark Details
When the benchmark is added, you can click "Run" button to execute it. Results will be filled in the table and in details view visible after clicking row in the table. You can also see config and output logs when clicking links highlighted in blue.

![Benchmark-details](imgs/bench/benchmark-details.png "Benchmark-details")


## Profiling Tab
### Profiling Table
It is also possible to do profiling of all Tensorflow frozen models in project.
![Profiling-table](imgs/bench/profiling-table.png "Profiling-table")
### Profiling Wizard
To profile model, click "Add new profiling" button at the bottom of the table and follow the steps.
![Profiling-wizard](imgs/bench/profiling-wizard.png "Profiling-wizard")

### Editing Profiling Entries
In Profiling tab you can edit dataset and number or threads. 
![Profiling-edit-wizard](imgs/bench/profiling-edit-wizard.png "Profiling-edit-wizard")


### Profiling Details
Once profiling entry is added, you can click "Run" button to execute it. After completing the process, the results will appear in the form of a bar chart and a table with full profiling data. The table is also used to control which operations are included in the chart. Check the box next to the selected row and click "Update chart" button to include it in the bar chart.
Click "Download .csv file" button to get profiling data in .csv file.
![Profiling-details](imgs/bench/profiling-details.png "Profiling-details")


# Diagnosis Tab
Diagnosis tab offers convenient debug information for optimizations with easy way for generating new one with requested variations.

To get OP list you need to execute quantization optimization and select optimized model on left hand side. In OP table you can see list of OPs with MSE and min/max activation values. Selecting one of OP in table highlights its position in graph. 

![Diagnosis-tab](imgs/bench/diagnosis-tab.jpg "Diagnosis-tab")

Configuration for currently selected OP can be set in section under graph and OP table. Activation and weights histograms can be generated in this section as shown in the picture below.
![Diagnosis-histogram](imgs/bench/histogram.jpg "Histogram")

You can set model wise parameters that apply to whole model by clicking button with "Model wise". When you set specific configuration you can view summary and generate new optimization config.

![Diagnosis-actions](imgs/bench/diagnosis-actions.png "Diagnosis-actions")

Model wise configuration provides separate settings for weights and activations.
![Diagnosis-model-wise-wizard](imgs/bench/diagnosis-model-wise-wizard.png "Diagnosis-model-wise-wizard")

## Dataset Tab
### Dataset List
Dataset tab presents list of datasets assigned to a project. In most cases the "dummy" dataset consisting of synthetic data should be automatically added while creating a project.
![Datasets-table](imgs/bench/datasets-table.png "Datasets-table")
### Dataset Wizard
New dataset can be defined by clicking "Add new profiling" button at the bottom of the table and follow the steps.
![Datasets-wizard](imgs/bench/datasets-wizard.png "Datasets-wizard")

### Dataset Details
Dataset details can be inspected by clicking specific row.
![Dataset-details](imgs/bench/dataset-details.png "Dataset-details")

### Custom Dataset
When adding the dataset, you can choose *custom* in dataloader and metric field. In that case a template file will be created. The path to the template file will be available in dataset details. You should edit this file to add your custom configuration before using this dataset in optimizations or benchmarks. Small yellow warning will remind about it. 

![Custom dataset](imgs/bench/custom-dataset.png "Custom dataset")

## Project Information

Last tab is called "Project info". You can find here details about the project, when it was created and modified, what is the framework and some details about input model. It is also possible to add some notes about the project.

![Project info](imgs/bench/project-info.png "Project info")

## NAS demo

Neural architecture search (NAS) example can be created from branch called [NAS_demo](https://github.com/intel/neural-compressor/tree/NAS_demo). The following dependencies must be installed before running the code:
```
pip install ofa fvcore torchvision autograd pymoo numba
```
Then follow the instructions from [introduction](#introduction).

To create new NAS example, click the `Create new project` button and in the second step called `Project type` choose `NAS example` like in the picture below.

![Project wizard2](imgs/bench/wizard2.png)

In third step called `Details` choose path to the [ImageNet ILSVRC dataset](https://www.image-net.org/).

![Project wizard3](imgs/bench/wizard3.png)

Details of the demo can be found in the initial view before running the demo:
![New NAS](imgs/bench/new-nas.png)

To run the demo click `Run` button. The process may take several hours. The results are shown in the chart:

![Executed NAS](imgs/bench/executed-nas.png)

## System Information

One can see system information by clicking ![System info](imgs/bench/system_info.png "System info") button. The result is details dialog:

![System info table](imgs/bench/system_info_table.png "System info table")


## Security
Intel® Neural Compressor Bench uses encrypted connections to ensure security, confidentiality and integrity of all client-server communication.

You can use automatically generated self-signed certificate or provide your own trusted certificate.

You can also choose to start the server without encryption exposing it to threats from network.

Intel® Neural Compressor Bench uses external packages to run the web-server and provide encryption. Please report any security issues to correct organizations:
- [Cryptography module](https://cryptography.io/en/latest/security/)
- [Flask](https://flask.palletsprojects.com/en/2.0.x/contributing/#reporting-issues)
- [Flask-SocketIO](https://github.com/miguelgrinberg/Flask-SocketIO/blob/main/SECURITY.md)
