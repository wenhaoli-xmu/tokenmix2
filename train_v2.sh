ENV_CONF=$1

deepspeed --include localhost:0 train_v2.py \
    --deepspeed_config ds_config_zero3.json \
    --env_conf $ENV_CONF
