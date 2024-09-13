#!/bin/bash
module load python/3.8.6
source ~/pfenv/bin/activate

for subdir in ls $1;
    do
        if [ -d "$1/$subdir" ]; then
            echo "Syncing $1/$subdir"
            wandb sync $1/$subdir --sync-all
        fi
    done
