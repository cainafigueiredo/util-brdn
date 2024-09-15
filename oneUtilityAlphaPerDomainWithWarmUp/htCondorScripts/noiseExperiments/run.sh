#!/bin/bash

experimentJSON=$1
experimentName=${experimentJSON%.*}

echo $experimentName
echo $experimentDir

tar -xzvf srlearn.tar.gz
tar -xzvf relatedWork.tar.gz
tar -xzvf utils.tar.gz
tar -xzvf data.tar.gz

tar -xzvf .venv.tar.gz

echo $(pwd) > _condor_stdout
output=$(ls -a -l)
printf "%s\n" "$output" >> _condor_stdout

chmod +x noiseExperiments_HTCondorVersion.py
./noiseExperiments_HTCondorVersion.py --experimentJSON $experimentJSON

experimentDir=$(ls -1 | grep "\-randomSeed_")
pushd $experimentDir
tar -czf $experimentName.tar.gz results
popd
mv $experimentDir/$experimentName.tar.gz .

echo $(tree) > _condor_stderr
