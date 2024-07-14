ENV_CONF=$1

deepspeed --include localhost:4,5,6,7 train.py \
    --deepspeed_config ds_config_zero3.json \
    --env_conf $ENV_CONF
