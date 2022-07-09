# poptorch-lstm-energy
energy demand lstm prediction on graphcore

## run interactive job
    srun -p ipu --gres=ipu:1 --cpus-per-task 128 --pty bash -i

## run batch job
modify job.batch

    bash run.sh

## singularity
    singularity shell docker://matejcvut/poptorch-dev
    singularity shell docker://graphcore/pytorch:latest

### Docker notes
    docker build -t poptorch-dev -f Dockerfile .
    docker run --rm -it --name graphcore -v $(pwd):/poptorch poptorch-dev:latest bash

