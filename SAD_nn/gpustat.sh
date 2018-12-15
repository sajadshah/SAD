#!/bin/bash

x=$(nvidia-smi --query-gpu=memory.free --format=noheader,csv | tr '\n' ',' | sed 's| %||g' | sed 's|,$||g')
echo $x
