#!/bin/bash
#!/bin/bash

# activate venv
source ../../.venv/bin/activate

# create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
data_pct=0.2
# timestamp for this batch
TS=$(date +"%Y-%m-%d_%H-%M-%S")

python ./train_WANDB.py --no-wandb --node-type waypoints --task feas_quantile \
  --percentiles 0.1 0.3 0.5 0.7 0.9 \
  --data-percentage $data_pct \
  --datadir ../data/randomBlocks_all/ \
  --project randomBlocks_all_$data_pct \
  > "$LOG_DIR/waypoints_feas_quantile_$TS.log" 2>&1

python ./train_WANDB.py --no-wandb --node-type waypoints --task infeas_quantile \
  --percentiles 0.1 0.3 0.5 0.7 0.9 \
  --data-percentage $data_pct \
  --datadir ../data/randomBlocks_all/ \
  --project randomBlocks_all_$data_pct \
  > "$LOG_DIR/waypoints_infeas_quantile_$TS.log" 2>&1

python ./train_WANDB.py --no-wandb --node-type waypoints --task feasibility \
  --percentiles 0.1 0.3 0.5 0.7 0.9 \
  --data-percentage $data_pct \
  --datadir ../data/randomBlocks_all/ \
  --project randomBlocks_all_$data_pct \
  > "$LOG_DIR/waypoints_feasibility_$TS.log" 2>&1

python ./train_WANDB.py --no-wandb --node-type rrt --task feas_quantile \
  --percentiles 0.1 0.3 0.5 0.7 0.9 \
  --data-percentage $data_pct \
  --datadir ../data/randomBlocks_all/ \
  --project randomBlocks_all_$data_pct \
  > "$LOG_DIR/rrt_feas_quantile_$TS.log" 2>&1

python ./train_WANDB.py --no-wandb --node-type lgp --task infeas_quantile \
  --percentiles 0.1 0.3 0.5 0.7 0.9 \
  --data-percentage $data_pct \
  --datadir ../data/randomBlocks_all/ \
  --project randomBlocks_all_$data_pct \
  > "$LOG_DIR/lgp_infeas_quantile_$TS.log" 2>&1

python ./train_WANDB.py --no-wandb --node-type lgp --task feasibility \
  --percentiles 0.1 0.3 0.5 0.7 0.9 \
  --data-percentage $data_pct \
  --datadir ../data/randomBlocks_all/ \
  --project randomBlocks_all_$data_pct \
  > "$LOG_DIR/lgp_feasibility_$TS.log" 2>&1

python ./train_WANDB.py --no-wandb --node-type lgp --task feas_quantile \
  --percentiles 0.1 0.3 0.5 0.7 0.9 \
  --data-percentage $data_pct \
  --datadir ../data/randomBlocks_all/ \
  --project randomBlocks_all_$data_pct \
  > "$LOG_DIR/lgp_feas_quantile_$TS.log" 2>&1
