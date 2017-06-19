#!/bin/bash

if [ ! -x /usr/bin/unzip ]; then
	echo "Cannot execute unzip binary, exiting..."
	exit -1
fi

if [ ! -e "./Sberbank Housing Market.zip" ]; then
	echo "Cannot find Sberbank zip file, exiting..."
	exit -2
fi

/usr/bin/unzip -q "./Sberbank Housing Market.zip"

for f in {macro.csv.zip,sample_submission.csv.zip,test.csv.zip,train.csv.zip}; do
	if [ ! -e "./$f" ]; then
		echo "Cannot find ./$f, exiting..."
		exit -3
	fi
	echo "Extracting ./${f:0:-4}"
	/usr/bin/unzip -q "./$f"

	rm "./$f"
done

echo "Finished! Cleaning up..."

rm -r "./__MACOSX"
