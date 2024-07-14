import os, sys
import json
from threading import Thread

import argparse, time
import psutil


def run_conf(conf):
    for config in conf:
        script = config['script']
        env_conf = config['env_conf']
        log = config['log']

        print("running script ...", flush=True)
        print(json.dumps(config, indent=4), flush=True)
        command = f"nohup python {script} --env_conf {env_conf} > {log} 2>&1"
        os.system(command)
        print("done!", flush=True)
        print("=" * 80, flush=True)


if __name__ == '__main__':
    with open("scripts/experiment.json", 'r') as f:
        conf = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--after", type=int, default=None)
    args = parser.parse_args()

    if args.after is not None:
        if psutil.pid_exists(args.after) is False:
            raise RuntimeError
        print(f"waiting for job {args.after}.", flush=True)
        while psutil.pid_exists(args.after) is True:
            time.sleep(1)

    num_groups = len(conf)
    threads = []
    for group_idx in range(num_groups):
        th = Thread(target=run_conf, args=(conf[group_idx],), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
