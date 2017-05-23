#!/bin/bash

# usage: source init.sh

export ORIGINAL_PS1=${PS1}
export PS1="${PS1:0: -5} (anaconda)\n$ "

export ORIGINAL_PATH=${PATH}                              # backup the original PATH
export PATH=/scratch/gmulder/anaconda2-4.3.1/bin:${PATH}  # add Anaconda2
