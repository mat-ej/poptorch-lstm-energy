# poptorch-lstm-energy
energy demand lstm prediction on graphcore.

## Installation / Local
build docker image

    docker build -t poptorch-dev -f Dockerfile .

run container in interactive mode

    docker run --rm -it --name graphcore -v $(pwd):/poptorch poptorch-dev:latest bash

use python3 to execute from /poptorch directory

    root@0f5fc525fc6e:/poptorch# python3 poptorch_energy/energy_lstm.py

## Cluser / IPU use in SLURM cluster

To set up the project inside slurm cluster, do one of the following:

### run interactive job
    srun -p ipu --gres=ipu:1 --cpus-per-task 128 --pty bash -i

### run batch job
modify job.batch accordingly

    bash run.sh


### singularity
    singularity shell docker://matejcvut/poptorch-dev
    singularity shell docker://graphcore/pytorch:latest


### Docker notes
    docker build -t poptorch-dev -f Dockerfile .
    docker run --rm -it --name graphcore -v $(pwd):/poptorch poptorch-dev:latest bash


## TODO
'poptorch_cpp_error': Failed to acquire 1 IPU(s)


