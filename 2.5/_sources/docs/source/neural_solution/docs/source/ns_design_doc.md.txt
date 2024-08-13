## Design Doc for Optimization as a Service [WIP]



### Contents

- [Design Doc for Optimization as a Service \[WIP\]](#design-doc-for-optimization-as-a-service-wip)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Workflow of OaaS](#workflow-of-oaas)
  - [Class definition diagram](#class-definition-diagram)
  - [Extensibility](#extensibility)

### Overview

Optimization as a service(OaaS) is a platform that enables users to submit quantization tasks for their models and automatically dispatches these tasks to one or multiple nodes for accuracy-aware tuning. OaaS is designed to parallelize the tuning process in two levels: tuning and model. At the tuning level, OaaS execute the tuning process across multiple nodes for one model. At the model level, OaaS allocate free nodes to incoming requests automatically.


### Workflow of OaaS

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
            Scheduler ->> Cluster: P2-2. Apply for resources
            Note over Scheduler, Cluster: the number of Nodes
            Cluster ->> Cluster: P2-3. Check the status of nodes in cluster
            Cluster ->> Scheduler: P2-4. Resources info
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
    Studio ->> ResultMonitor: P4-1. Query the status of the submitted task
    ResultMonitor ->> Studio: P4-2. Post the status of queried task
    End

```

The optimization process is divided into four parts, each executed in separate threads.

- Part 1. Posting new quantization task. (P1-1 -> P1-2 -> P1-3)

- Part 2. Resource allocation and scheduling. (P2-1 -> P2-2 -> P2-3 -> P2-4 -> P2-5)

- Part 3. Task execution and reporting. (P3-1 -> P3-2 -> P3-3 -> P3-4 -> P3-5)

- Part 4. Updating the status. (P4-1 -> P4-2)

### Class definition diagram



```mermaid
classDiagram



TaskDB "1" --> "*" Task
TaskMonitor --> TaskDB
ResultMonitor  -->  TaskDB
Scheduler --> TaskDB
Scheduler --> Cluster


class Task{
	+ status
	+ get_status()
	+ update_status()
}

class TaskDB{
   - task_collections
   + append_task()
   + get_all_pending_tasks()
   + update_task_status()
}
class TaskMonitor{
    - task_db
    + wait_new_task()
}
class Scheduler{
    - task_db
    - cluster
    + schedule_tasks()
    + dispatch_task()
    + launch_task()
}

class ResultMonitor{
	- task_db
    + query_task_status()
}
class Cluster{
    - node_list
    + free()
    + reserve_resource()
    + get_node_status()
}

```


### Extensibility

- The service can be deployed on various resource pool, including a set of worker nodes, such as a local cluster or cloud cluster (AWS and GCP).
