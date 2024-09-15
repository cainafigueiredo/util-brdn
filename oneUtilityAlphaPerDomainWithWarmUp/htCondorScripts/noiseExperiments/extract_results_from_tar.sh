#!/bin/bash

if [ -z "$1" ]; then
	workDir="."
else
	workDir=$1
fi	

pushd $workDir

for file in $(ls -1 *.tar.gz); do
	experimentDir=$(tar -tzf $file | grep -E 'results/[^/]+/?$')
	experimentDir=./${experimentDir%/}
	echo "Extracting $file to $experimentDir..."
	tar -xzf $file
	rm $file
done

popd
