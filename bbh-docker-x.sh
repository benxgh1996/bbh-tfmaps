#!/bin/bash -x
docker run \
    -it -u $(id -u):$(id -g) \
    --name bbh-tf-session \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /Users/benxgh1996/desktop/rsch/bbh-tfmaps:/bbh-tfmaps \
    -v /Users/benxgh1996/desktop/rsch/bbh-tfmaps/pyutils:/pyutils \
    -v /Users/benxgh1996/desktop/rsch/lvcnr-lfs/GeorgiaTech:/waves \
    -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
    -w /bbh-tfmaps -p 8888:8888 jclarkastro/bbh-tfmaps