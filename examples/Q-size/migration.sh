#!/bin/bash
if [[ "$1" = "routenet" ]]; then

    python3 migrate_routenet_example.py  --dataset  $2 --output_path $3

fi

if [[ "$1" = "q-size" ]]; then

    python3 migrate_q_size_example.py  --dataset  $2 --output_path $3

fi