#!/bin/bash

docker run --runtime nvidia -it --rm \
    -v $(pwd)/../:/home/$USER \
    transfusion \
    /bin/bash -c "cd /home/$USER; exec /bin/bash"
