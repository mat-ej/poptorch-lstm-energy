# poptorch-lstm-energy
energy demand lstm predicition on graphcore

docker run --rm -it --name graphcore -v $(pwd):/poptorch poptorch-dev:latest bash
docker build -t poptorch-dev -f Dockerfile .