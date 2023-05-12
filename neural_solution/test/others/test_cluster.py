from backend.cluster import Node, Cluster


node1 = Node(name='mlt-skx121',num_sockets=2, num_cores_per_socket=20)
node2 = Node(name='mlt-skx122',num_sockets=2, num_cores_per_socket=20)
node3 = Node(name='mlt-skx124',num_sockets=2, num_cores_per_socket=20)

node_lst = [node1, node2, node3]
cluster = Cluster(node_lst=node_lst)

from backend.task import Task
task1 = Task(task_id = 101, arguments="ls", unit_num=3, status='pending')
task2 = Task(task_id = 102, arguments="ls", unit_num=2, status='pending')
task3 = Task(task_id = 103, arguments="ls", unit_num=3, status='pending')
task4 = Task(task_id = 104, arguments="ls", unit_num=1, status='pending')
cluster.reserve_resource(task=task1)
cluster.reserve_resource(task=task2)
cluster.reserve_resource(task=task3)
cluster.reserve_resource(task=task4)

