import argparse

parser = argparse.ArgumentParser(description = "Prepare experiment files.")

parser.add_argument("--experimentsJSON", type = str, help = "Path to a raw JSON file specifying the experiments.", required = True)
parser.add_argument("--database", type = str, help = "Database name.", required = True)
parser.add_argument("--beta", type = float, help = "Balance strength used by the Instance Group Soft Balance weight strategy.", required = True)
parser.add_argument("--randomSeed", type = int, help = "Random seed.", required = True)
parser.add_argument("--outputDir", type = str, help = "Output directory.", required = True)
parser.add_argument("--finishedExperimentsTXT", type = str, help = "Path to a .txt file listing all finished experiment identifiers, one id per line. These experiments are skipped.", required = False, default = "")

args = parser.parse_args()

experimentsJSONFilePath = args.experimentsJSON
databaseFilter = args.database
beta = args.beta
randomSeed = args.randomSeed
outputDir = args.outputDir
jsonDir = f"{outputDir}/json"
resultsDir = f"{outputDir}/results"

finishedExperimentsIDList = []
if args.finishedExperimentsTXT != "":
    with open(args.finishedExperimentsTXT) as f:
        finishedExperimentsIDList = f.read().strip().split("\n")
finishedExperimentsIDList = set(finishedExperimentsIDList)

import os

import json
import re

dataDir = "./myProject/transferLearning/data"

# This function can be leveraged to prioritize experiments.
def skipExperiment(experimentDict):
    # The experiment has already been carried out.
    experimentID = experimentDict["id"]
    experimentPath = experimentDict["path"]
    
    if os.path.exists(f"{experimentPath}/{experimentID}/metrics.json"):
        with open(f"{experimentPath}/{experimentID}/metrics.json") as f:
            if f.read() != "{}":
                return True, "The experiment has already been carried out."
    
    datasetPath = experimentDict.get("databasePath")
    datasetName = os.path.basename(datasetPath)

    if datasetName != databaseFilter:
        return True, f"Database is not in {databaseFilter}"

    weightStrategy = experimentDict["weight"]["strategy"]
    if weightStrategy != "balancedInstanceGroupUniform":
        return True, f"Weight strategy is different than 'balancedInstanceGroupUniform'"
    else:
        weightParameters = experimentDict["weight"]["parameters"]
        balanceStrength = weightParameters["balanceStrength"]
        if balanceStrength != beta:
            return True, f"`balanceStrength` is different than {beta}"

    if experimentDict["randomSeed"] != randomSeed:
        return True, f"`randomSeed` is different than {randomSeed}"

    if experimentID in finishedExperimentsIDList:
        return True, f"Experiment {experimentID} has already been carried out, according to {args.finishedExperimentsTXT}."

    return False, ""

# Loading raw experiment files
experiments = {}
with open(experimentsJSONFilePath) as f:
    experiments = json.load(f)

experimentsToRun = []
for experimentDict in experiments:
    experimentID = experimentDict['id']
    experimentDict["path"] = resultsDir
    experimentDict["databasePath"] = re.sub(r".*/data", dataDir, experimentDict["databasePath"])
    shouldSkipExperiment, skipMessage = skipExperiment(experimentDict)
    if not shouldSkipExperiment:
        experimentsToRun.append(experimentDict)

# Sorting experiments by noiseStrength
experimentsToRun = sorted(
    experimentsToRun, 
    key = lambda expDict: (
        expDict["noiseStrength"], expDict["sourceUtilityAlpha"], expDict["targetUtilityAlpha"]
    )
)

os.makedirs(jsonDir, exist_ok = True)

for i, experimentDict in enumerate(experimentsToRun):
    with open(f"{jsonDir}/experiment_{i}.json", "w") as f:
        json.dump(experimentDict, f)
