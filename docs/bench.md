Intel® Neural Compressor Bench
=======

Intel® Neural Compressor Bench is a web application for easier use of Intel® Neural Compressor. It is only available on Linux based hosts.

## Table of Contents
- [Introduction](#introduction)
- [Home screen](#home-screen)
- [Create new project](#create-new-project)
  - [Add optimization](#add-optimization)
  - [Add benchmark](#add-benchmark)
  - [Add profiling](#add-profiling)
  - [Display model graph](#display-model-graph)
  - [Add dataset](#add-dataset)
  - [Project information](#project-information)
- [System information](#system-information)
- [Security](#security)
# Introduction
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

## Home screen
This view shows introduction to Intel® Neural Compressor Bench and a button for creating new project.

![Home](imgs/bench/home.png "Home")

# Create new project
To create a new project, you need to choose its name and input model.

![Project1](imgs/bench/project1.png "Project1")

When the model is chosen, you can also edit its input and output nodes, see the model graph (for Tensorflow models) and set shape for synthetic dataset. It is also possible that you will have to choose model domain if it was not auto detected. It is used to set some default parameters.

![Project2](imgs/bench/project2.png "Project2")

On the left hand side there is a panel with list of created projects. When  you click on the project name, you can see its details. "Create new project" button goes to new project wizard described in previous section.

![Menu](imgs/bench/menu.png "Menu")

## Add optimization
### Optimization table
In Optimizations tab you can see list of optimizations in the project. Currently UI supports three optimization precisions and two types of optimization.
![Optimizations-table](imgs/bench/optimizations-table.png "Optimizations-table")

### Optimization wizard
To add new optimization, click "Add new optimization" button at the bottom of the table and follow the steps.
![Optimizations-wizard](imgs/bench/optimizations-wizard.png "Optimizations-wizard")
### Optimization details
To perform optimization click "Run" button. Once process is finished you can click on row with specific optimization to display details about optimization parameters and optimized model.
![Optimization-details](imgs/bench/optimization-details.png "Optimization-details")

## Add benchmark
### Benchmark table
For each optimization and input model you can add benchmark. Benchmark have 2 modes: accuracy and performance. In benchmark tab you can see all your benchmarks. When you check checkboxes in the last column you can choose benchmark you want to compare in the chart (visible after clicking "Compare selected").

![Benchmarks-table](imgs/bench/benchmarks-table.png "Benchmarks-table")

### Benchmark wizard
To add new benchmark, click "Add new benchmark" button at the bottom of the table and follow the steps.

![Benchmarks-wizard](imgs/bench/benchmarks-wizard.png "Benchmarks-wizard")

### Benchmark details
When the benchmark is added, you can click "Run" button to execute it. Results will be filled in the table and in details view visible after clicking row in the table. You can also see config and output logs when clicking links highlighted in blue.

![Benchmark-details](imgs/bench/benchmark-details.png "Benchmark-details")


## Add profiling
### Profiling table
It is also possible to do profiling of all Tensorflow frozen models in project.
![Profiling-table](imgs/bench/profiling-table.png "Profiling-table")
### Profiling wizard
To profile model, click "Add new profiling" button at the bottom of the table and follow the steps.
![Profiling-wizard](imgs/bench/profiling-wizard.png "Profiling-wizard")
### Profiling details
Once profiling entry is added, you can click "Run" button to execute it. After completing the process, the results will appear in the form of a bar chart and a table with full profiling data. The table is also used to control which operations are to be included in the chart. Check the box next to the selected row and click "Update chart" button to include it in the bar chart.
![Profiling-details](imgs/bench/profiling-details.png "Profiling-details")

## Display model graph
For Tensorflow frozen pb models there will be a button available ![Show graph](imgs/bench/show_graph_button.png "Show graph") in the project wizard. It is also possible to see the graph in graph tab. The graph by default is collapsed, but when you click on plus icon, sections will be unfolded.


![Bert model graph](imgs/bench/graph_bert.png "Bert model graph").

## Add dataset
### Dataset list
Dataset tab presents list of datasets assigned to a project. In most cases the "dummy" dataset consisting of synthetic data should be automatically added while creating a project.
![Datasets-table](imgs/bench/datasets-table.png "Datasets-table")
### Dataset wizard
New dataset can be defined by clicking "Add new profiling" button at the bottom of the table and follow the steps.
![Datasets-wizard](imgs/bench/datasets-wizard.png "Datasets-wizard")

### Dataset details
Dataset details can be inspected by clicking specific row.
![Dataset-details](imgs/bench/dataset-details.png "Dataset-details")
## Project information

Last tab is called "Project info". You can find here details about the project, when it was created and modified, what is the framework and some details about input model. It is also possible to add some notes about the project.

![Project info](imgs/bench/project-info.png "Project info")

## System information

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
