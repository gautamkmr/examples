#Use the below script and run it on each machine with only change on Node value

```
#!/bin/bash

export MASTER_ADDR=<ip address of master>
export MASTER_PORT=<port>
export WORLD_SIZE=<total number of gpu>

kill_children() {
  for PID in ${PIDS[*]}; do
    kill -TERM $PID
  done
}

NODE=0 #Change this for each machine indexed with 0, 1, 2 ... N
RANKS_PER_NODE=8 #For P3.16x machine 


for i in $(seq 0 7); do
  LOCAL_RANK=$i
  RANK=$((RANKS_PER_NODE * NODE + LOCAL_RANK))
  NCCL_DEBUG=INFO NCCL_MIN_NRINGS=5 python /home/ubuntu/examples/imagenet/main.py  \
       --a resnet50 \
       /home/ubuntu/imagenet \
       --dist-url env://        \
       --gpu $LOCAL_RANK \
       -j 4 \
       --epochs 2 \
       --batch-size 32 \
       --dist-backend nccl &
  PIDS[$LOCAL_RANK]=$!
done


trap kill_children SIGTERM SIGINT

for PID in ${PIDS[*]}; do
  wait $PID
done
```
