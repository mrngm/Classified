#!/bin/bash

# usage: source reset.sh

export PS1=${ORIGINAL_PS1}

export PATH=${ORIGINAL_PATH}

echo "Restoring $HOME/.keras/keras.json backup"
mv -v ${KERASBACKUPFN} "$HOME/.keras/keras.json"

unset KERASBACKUPFN
