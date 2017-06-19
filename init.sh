#!/bin/bash

# usage: source init.sh
# it expects Anaconda2 to reside in '../anaconda2-4.3.1/'

export ORIGINAL_PS1=${PS1}
export PS1="${PS1:0: -5} (anaconda)\n$ "

export ORIGINAL_PATH=${PATH}                   # backup the original PATH
export PATH="`pwd`/../anaconda2-4.3.1/bin:${PATH}"      # add Anaconda2

if [ -f "$HOME/.keras/keras.json" ]; then
	echo "Backup up $HOME/.keras/keras.json and replacing it with our own"
	export KERASBACKUPFN="$HOME/.keras/keras.json.`date +%s`.backup"
	cp "$HOME/.keras/keras.json" "$KERASBACKUPFN"
	cat <<EOF >"$HOME/.keras/keras.json"
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "theano",
    "image_data_format": "channels_last"
}
EOF
fi
