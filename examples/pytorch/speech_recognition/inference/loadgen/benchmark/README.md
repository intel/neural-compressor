Note: please install jemalloc first. See: http://jemalloc.net/
Command: bash run.sh <target_qps> <0=Basic,1=Queue> <numCompleteThreads> <maxSizeInComplete> <server_coalesce_queries=0or1>

Experiments:
- On Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz
- Basic SUT : 500-600k i/s
- Basic SUT + jemalloc: 800-900k i/s (`bash run.sh 800000 0`)
- Queued SUT (2 complete threads) + jemalloc: 1.2-1.3M i/s (`bash run.sh 1200000 1 2 2048`)
- Queued SUT (2 complete threads) + jemalloc + server_coalesce_queries: 1.4-1.5M is/ (`bash run.sh 1400000 1 2 512 1`)
- Basic SUT + jemalloc + server_coalesce_queries + 4 IssueQueryThreads: 2.4-2.5M is/ (`bash run.sh 2400000 0 2 512 1 4`)
