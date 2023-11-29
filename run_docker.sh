#!/bin/bash
docker run --gpus all --rm -ti -v /PATH/TO/DATA:/Data --ipc=host projectmonai/monai:1.1.0 
