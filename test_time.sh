#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

measure_runtime() {
    local start_time=$(date +%s)
    python pred.py "$@"
    local end_time=$(date +%s)
    echo $((end_time - start_time))
}

# duration_longchat=$(measure_runtime --env_conf longchat-7b.json --model_max_length 32768 --max_gen 0)
# duration_longalpaca=$(measure_runtime --env_conf longalpaca-7b.json --model_max_length 16384 --max_gen 0)
# duration_yarn=$(measure_runtime --env_conf yarn-7b.json --model_max_length 32768 --max_gen 0)
# duration_beacon=$(measure_runtime --env_conf beacon.json --model_max_length 32768 --max_gen 0)
duration_ours=$(measure_runtime --env_conf hybird9-8.40000.json --model_max_length 32768 --chat_template vicuna_v1.1 --max_gen 0)

# echo "LongChat-7B runtime: $duration_longchat s"
# echo "LongAlpaca-7B runtime: $duration_longalpaca s"
# echo "yarn-7B runtime: $duration_yarn s"
# echo "Activation Beacon runtime: $duration_beacon s"
echo "Ours runtime: $duration_ours s"