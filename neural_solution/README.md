# What's Neural Solution?
<!-- TODO what is ns -->
Neural Solution is a flexible and easy to use tool that brings the capabilities of INC as a service. Users can effortlessly submit optimization tasks through the HTTP/gRPC API. Neural Solution automatically dispatches these tasks to one or multiple nodes, streamlining the entire process. 

# Why Neural Solution?
<!-- TODO what does the ns provide -->
- Efficient: Neural Solution speed up the optimization process by seamlessly parallelizes the tuning process across multi nodes.
- Security:
- Code Less:


# Get Started
<!-- TODO how to install it -->
## Installation
<details>
  <summary>Prerequisites</summary>

<!--TODO: Precise OS versions-->

- Operating systems
  - Linux
- Python: 3.8 ~ 3.10 <!--TODO: double check the PY version with the mpi4py support>
- Conda: ? <!--TODO: Precise Conda versions>
</details>

### Use conda to install neural solution:
```
conda install <!--TODO: align it with neural insights>
```

### Build from source:
<!--TODO: align it with neural insights>


# E2E examples
<!-- TODO highlights E2E examples -->
- Quantizing a Hugging Face model
- Quantizing a custom model
# Learn More
<!-- TODO more docs(Install details, API and so on...) -->

- The Architecture documents
- API Reference

# Contact

Please contact us at [inc.maintainers@intel.com](mailto:inc.maintainers@intel.com) for any Neural Coder related question.
<!-- TODO removed the content below 

# Optimization as a Service(OaaS)

Optimization as a Service(OaaS) is a platform for users to submit quantization/optimization tasks via HTTP/gRPC API. It dispatches tasks automatically to one or multiple nodes. 

### Software Architecture
![Software Architecture ](./docs/imgs/OaaS-Intro.png "Software Architecture")
### Workflow

```mermaid
sequenceDiagram
    participant Studio
    participant TaskMonitor
    participant Scheduler
    participant Cluster
    participant TaskLauncher
    participant ResultMonitor
    Par receive task
    Studio ->> TaskMonitor: P1-1. Post quantization Request
    TaskMonitor ->> TaskMonitor: P1-2. Add task to task DB
    TaskMonitor ->> Studio: P1-3. Task received notification
    and Schedule task
        loop 
            Scheduler ->> Scheduler: P2-1. Pop task from task DB
            Scheduler ->> Cluster: P2-2. Apply for resouces
            Note over Scheduler, Cluster: the number of Nodes
            Cluster ->> Cluster: P2-3. Check the status of nodes in cluster
            Cluster ->> Scheduler: P2-4. Resouces info 
            Note over Scheduler, Cluster: host:socket list
            Scheduler ->> TaskLauncher: P2-5. Dispatch task
        end
    and Run task
    TaskLauncher ->> TaskLauncher: P3-1. Run task
    Note over TaskLauncher, TaskLauncher: mpirun -np 4 -hostfile hostfile python main.py
    TaskLauncher ->> TaskLauncher: P3-2. Wait task to finish...
    TaskLauncher ->> Cluster: P3-3. Free resource
    TaskLauncher ->> ResultMonitor: P3-4. Report the Acc and Perf
    ResultMonitor ->> Studio: P3-5. Post result to Studio
    and Query task status
    Studio ->> ResultMonitor: P4-1. Query the status of the submmited task 
    ResultMonitor ->> Studio: P4-2. Post the status of queried task
    End

```

-->
