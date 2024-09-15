#!/usr/bin/env bash

scriptDir=$(readlink -f $0 | xargs dirname)
source $scriptDir/.env

database=$1
beta=$2
randomSeed=$3
finishedExperimentsTXT=$4
venvTarGzPath=$VENV_TAR_GZ_PATH
srlearnTarGzPath=$SRLEARN_TAR_GZ_PATH
relatedWorkTarGzPath=$RELATED_WORK_TAR_GZ_PATH
utilsTarGzPath=$UTILS_TAR_GZ_PATH
dataTarGzPath=${DATA_TAR_GZ_PATH}/${database}/data.tar.gz
experimentsJSONPath="$PROJECT_ROOT_PATH/oneUtilityAlphaPerDomainWithWarmUp/experimentSettingJSONs/noisyTransferLearning-only-$database/experiments-noisyTransferLearning-randomSeed=$randomSeed.json"

experimentsDir="./$database-beta_$beta-randomSeed_$randomSeed"
outputDir="$experimentsDir/output"
errorDir="$experimentsDir/error"
logDir="$experimentsDir/log"
jsonDir="$experimentsDir/json"
resultsDir="$experimentsDir/results"
submitFilePath="$experimentsDir/submit.sub"

if [ -d "$experimentsDir" ]; then
    echo "An directory for these experiments already exists on $experimentsDir."
    
    while true; do
        read -p "Would you like to update the list of pending experiments? (Y/n): " choice
	choice=${choice:-y}
        case $choice in
            [Yy]* )
                echo "Updating list of pending experiments..."
                rm -rf $outputDir $logDir $errorDir $jsonDir $submitFilePath
		break;;
            [Nn]* )
                echo "The generation has been canceled"
                exit;;
            * )
		echo "Please, answer with yes (y) or no (n).";;
        esac
    done
fi

tar -xzf $venvTarGzPath
source .venv/bin/activate

command="python3 $scriptDir/prepareExperiments.py --experimentsJSON $experimentsJSONPath --database $database --beta $beta --randomSeed $randomSeed --outputDir $experimentsDir"

if [ -f "$finishedExperimentsTXT" ]; then
  command="$command --finishedExperimentsTXT $finishedExperimentsTXT"
fi

eval $command

rm -rf .venv

mkdir -p $outputDir
mkdir -p $logDir
mkdir -p $errorDir
mkdir -p $resultsDir

totalExperiments=$(ls -1 $jsonDir | wc -l)
cat $scriptDir/template.sub | sed "s/<DATABASE>/$database/g" | sed "s/<BETA>/$beta/g" | sed "s/<RANDOM_SEED>/$randomSeed/g" | sed "s/<TOTAL_EXPERIMENTS>/$totalExperiments/g" | sed "s|<VENV_TAR_GZ_PATH>|$venvTarGzPath|g" | sed "s|<SRLEARN_TAR_GZ_PATH>|$srlearnTarGzPath|g" | sed "s|<RELATED_WORK_TAR_GZ_PATH>|$relatedWorkTarGzPath|g" | sed "s|<UTILS_TAR_GZ_PATH>|$utilsTarGzPath|g" | sed "s|<RUN_SCRIPT_DIR>|$scriptDir|g" | sed "s|<DATA_TAR_GZ_PATH>|$dataTarGzPath|g" > $submitFilePath

echo "$totalExperiments were genereated."
