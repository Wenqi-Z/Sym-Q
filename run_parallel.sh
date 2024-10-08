#!/bin/bash

# Number of parallel jobs
NUM_JOBS=10
MAX_BATCH=19 # 9999


python add_points_to_h5.py --mode val
seq 0 $MAX_BATCH| xargs -n 1 -P $NUM_JOBS -I {} python add_points_to_h5.py --batch {} --mode train
