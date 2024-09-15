#!.venv/bin/python3

import argparse
parser = argparse.ArgumentParser(description="Runs 'transfer from noisy source' experiments as specified in a JSON file.")
parser.add_argument("--experimentJSON", type = str, help = "Path to a JSON file specifying the experiment.", required = True)
args = parser.parse_args()

import os
experimentJSONPath = args.experimentJSON

import sys
# PROJECT_PATH = os.environ.get("PROJECT_ROOT_PATH")
# SRLEARN_PATH = os.environ.get("SRLEARN_ROOT_PATH")
# sys.path.append(SRLEARN_PATH)
sys.path.append("./myProject/srlearn")
# sys.path.append(PROJECT_PATH)
sys.path.append("./myProject/transferLearning")

import re
import json
import functools
import numpy as np
import networkx as nx

from glob import glob
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from srlearn.database import Database
from srlearn.weight import WeightFactory

from utils.experiment import loadDatabase, getLogger, runSingleExperiment_TransferLearning

from concurrent.futures import ProcessPoolExecutor
import multiprocessing.managers

import warnings
warnings.filterwarnings("ignore")

def addGraphs(G1, G2):
    # This method assumes that:
    #   - V1 (set of nodes from G1) is a subset of V2 (set of nodes from G2)
    #   - Each node has the same id in both graphs
    #   - Each edge has an attribute named `edgeType`
    # It is inpired by the following paper: 
    #   - Graph Perturbation as Noise Graph Addition: A New Perspective for Graph Anonymization (https://link.springer.com/chapter/10.1007/978-3-030-31500-9_8)
    #       - Adding edges: an edge from G1 does not exist in G2
    #       - Edges removal: an edge from G1 shares the same type as the corresponding edge from G2
    #       - Edge type modification: an edge type from G1 is different from the corresponding edge in G2 
    # Due to Edge type modification, (G1 + G2) != (G2 + G1)

    G = G2.copy()

    for edge in G1.edges:
        # Adding edges
        if edge not in G2.edges:
            edgeType = G1.edges[edge]["edgeType"]
            G.add_edge(*edge, edgeType = edgeType)
        
        elif edge in G2.edges:
            edgeTypeG1 = G1.edges[edge]["edgeType"]
            edgeTypeG2 = G2.edges[edge]["edgeType"]
            
            # Edges removal
            if edgeTypeG1 == edgeTypeG2:
                G.remove_edge(*edge)
            
            # Edge type modification
            else:
                G.edges[edge]["edgeType"] = edgeTypeG1

    return G

def experimentResultSummarization(experimentResult: dict, logger = None, ):
    if not logger:
        logger = getLogger("Result summarization")

    logger.info("Extracting performance metrics from experiment results:")
    metrics = {}
    for exp, expResults in experimentResult.items():
        metrics[exp] = metrics.get(exp, {})
        for trainFold, foldResults in expResults.items():
            for metricName, metricValue in foldResults["metrics"].items():
                metrics[exp][metricName] = metrics[exp].get(metricName, [])
                metrics[exp][metricName] += [float(metricValue)]

    for exp, expMetrics in metrics.items():
        for metricName, metricValues in expMetrics.items():
            metricValues = np.array(metricValues)
            mean = metricValues.mean()
            std = metricValues.std()
            logger.info(f"{exp}: {metricName} = {mean:.4f} +- {std:.4f}")

def runNoiseExperiment(
    experimentDict: dict, logger = None
):
    experiment = experimentDict
    experimentID = experiment["id"]
    experimentBasePath = os.path.join(experiment["path"], experimentID)

    os.makedirs(experimentBasePath, exist_ok = True)

    with open(os.path.join(experimentBasePath, "setting.json"), "w") as f:
        json.dump(experiment, f)

    if not logger:
        logger = getLogger(experimentID, level = logging.DEBUG)

    logger.info("Parsing experiment parameters...")

    randomSeed = experiment.get("randomSeed")
    useRecursion = experiment.get("useRecursion", False)
    negPosRatio = experiment.get("negPosRatio", 1)
    maxFailedNegSamplingRetries = experiment.get("maxFailedNegSamplingRetries", 50)
    nEstimators = experiment.get("nEstimators", 10)
    nodeSize = experiment.get("nodeSize", 2)
    maxTreeDepth = experiment.get("maxTreeDepth", 3)
    numberOfClauses = experiment.get("numberOfClauses", 8)
    numberOfCycles = experiment.get("numberOfCycles", 100)
    ignoreSTDOUT = experiment.get("ignoreSTDOUT", True)
    weightFactory = WeightFactory()
    weightStrategy = weightFactory.getWeightStrategy(
        experiment["weight"]["strategy"], 
        **experiment["weight"]["parameters"]
    )
    sourceUtilityAlpha = experiment.get("sourceUtilityAlpha", 1)
    targetUtilityAlpha = experiment.get("targetUtilityAlpha", 1)
    utilityAlphaSetIter = experiment.get("utilityAlphaSetIter", 1)

    trainNSplits = experiment.get("trainNSplits", 5) # The train set is split into folds. Some of them are used as source set and the remaing are used as target set.
    trainSourceSplits = experiment.get("trainSourceSplits", trainNSplits - 1) # Number of train folds to be used as source. 
    trainTargetSplits = trainNSplits - trainSourceSplits # Number of train folds to be used as target.
    noiseStrength = experiment.get("noiseStrength", 0.005) # Probability of adding an edge between a given pair of nodes

    datasetPath = experiment.get("databasePath")
    targetPredicate = experiment.get("targetPredicate", None)
    resetTargetPredicate = experiment.get("resetTargetPredicate", None)
    datasetFolds = [os.path.basename(foldPath) for foldPath in glob(f"{datasetPath}/fold*")]

    np.random.seed(randomSeed)
    result = {}
    for testFold in datasetFolds:
        experimentFoldPath = os.path.join(experimentBasePath, testFold)
        os.makedirs(experimentFoldPath, exist_ok = True)

        logger.info(f"RUNNING EXPERIMENTS USING {testFold.upper()} AS TEST FOLD...")

        logger.info(f"Loading test database and converting it to arity 2...")
        testDataset = loadDatabase(
            path = datasetPath,
            folds = [testFold],
            useRecursion = useRecursion,
            targetPredicate = targetPredicate,
            resetTargetPredicate = resetTargetPredicate, 
            negPosRatio = negPosRatio,
            maxFailedNegSamplingRetries = maxFailedNegSamplingRetries,
            logger = logger
        )
        testDataset = Database.convertDatabaseToArity2(testDataset)

        logger.info("Loading database for training and converting it to arity 2...")

        trainFolds = [fold for fold in datasetFolds if fold != testFold]
        trainDataset = loadDatabase(
            path = datasetPath,
            folds = trainFolds,
            useRecursion = useRecursion,
            targetPredicate = targetPredicate,
            resetTargetPredicate = resetTargetPredicate, 
            negPosRatio = negPosRatio,
            maxFailedNegSamplingRetries = maxFailedNegSamplingRetries,
            logger = logger
        )
        trainDataset = Database.convertDatabaseToArity2(trainDataset)

        logger.debug(f"Splitting train database into source {trainSourceSplits/trainNSplits*100:.0f}% and target {trainTargetSplits/trainNSplits*100:.0f}%...")

        trainDatasetSplits = list(Database.getKFolds(trainDataset, numFolds = trainNSplits, shuffle = True))
        trainDatasetTarget = functools.reduce(lambda merged, split: merged.merge(split), trainDatasetSplits[:trainTargetSplits])
        trainDatasetSource = functools.reduce(lambda merged, split: merged.merge(split), trainDatasetSplits[trainTargetSplits:])
        schema = trainDatasetSource.extractSchemaPreds()

        candidateRelationTypes = {} 
        for relation, nodeTypes in schema.items():
            sortedNodeTypes = tuple(sorted(nodeTypes))
            candidates = candidateRelationTypes.get(sortedNodeTypes, [])
            candidateRelationTypes[sortedNodeTypes] = candidates + [relation]

        trainGraphSource = Database.convertDatabaseToUndirectedGraph(trainDatasetSource)

        logger.debug(f"Adding noise to source database (noiseStrength: {noiseStrength})...")

        noiseGraph = nx.Graph()
        noiseGraph.add_nodes_from(trainGraphSource.nodes(data = True))

        potentialEdges = combinations(trainGraphSource.nodes, 2)

        for edge in potentialEdges:
            if np.random.rand() < noiseStrength:
                nodeA, nodeB = edge
                nodeAType = noiseGraph.nodes[nodeA]["nodeType"]
                nodeBType = noiseGraph.nodes[nodeB]["nodeType"]
                sortedNodeTypes = tuple(sorted([nodeAType, nodeBType]))
                relationCandidates = candidateRelationTypes.get(sortedNodeTypes)
                if relationCandidates:
                    sampledCandidate = np.random.choice(relationCandidates, 1)[0]
                    noiseGraph.add_edge(*edge, edgeType = sampledCandidate)

        logger.debug(f"Combining target and noisy source databases...")

        noiseTrainGraphSource = addGraphs(noiseGraph, trainGraphSource)

        noiseTrainDatasetSource = Database.populateFromGraph(
            graph = noiseTrainGraphSource,
            modes = trainDatasetSource.modes,
            targetRelation = trainDatasetSource.getTargetRelation(),
            useRecursion = useRecursion,
            negPosRatio = negPosRatio
        )

        relationMapping = {}
        termTypeMapping = {}
        for relationType, termTypes in schema.items():
            relationMapping[relationType] = relationType
            for termType in termTypes:
                termTypeMapping[termType] = termType

        logger.debug(f"Training and evaluating model...")

        result[testFold] = runSingleExperiment_TransferLearning(
            experimentPath = experimentFoldPath, 
            sourceDatabase = noiseTrainDatasetSource,
            targetDatabaseTrain = trainDatasetTarget,
            targetDatabaseTest = testDataset,
            nEstimators = nEstimators,
            nodeSize = nodeSize,
            maxTreeDepth = maxTreeDepth,
            negPosRatio = negPosRatio,
            numberOfClauses = numberOfClauses,
            numberOfCycles = numberOfCycles,
            ignoreSTDOUT = ignoreSTDOUT,
            useRecursion = useRecursion,
            randomSeed = randomSeed,
            maxFailedNegSamplingRetries = maxFailedNegSamplingRetries,
            weightStrategy = weightStrategy,
            sourceUtilityAlpha = sourceUtilityAlpha,
            targetUtilityAlpha = targetUtilityAlpha,
            utilityAlphaSetIter = utilityAlphaSetIter,
            relationMapping = relationMapping,
            termTypeMapping = termTypeMapping,
            logger = logger
        )

    metricsJSONPath = os.path.join(experimentBasePath, "metrics.json")
    logger.info(f"Storing performance metrics at {metricsJSONPath}.")

    allMetrics = {}
    for fold, foldResults in result.items():
        allMetrics[fold] = foldResults["metrics"]

    with open(metricsJSONPath, "w") as f:
        json.dump(allMetrics, f)

    logger.info("Experiment has been finished.")
        
    return {"transferLearning": result}

# Loading experiments JSON
experimentDict = {}
with open(experimentJSONPath) as f:
    experimentDict = json.load(f)

experimentID = experimentDict["id"]
os.system(f"rm -rf {experimentDict['path']}/{experimentID}")
print(f"Starting experiment {experimentID}...")
try:
    experimentPath = f"{experimentDict['path']}/{experimentID}"
    os.makedirs(experimentPath, exist_ok = True)    
    logger = getLogger(experimentID, logFile = f"{experimentPath}/experiment.log", consoleOutput = False)
    experimentResult = runNoiseExperiment(experimentDict, logger = logger)
    experimentResultSummarization(experimentResult, logger = logger)
    print(f"Experiment finished successfully: {experimentID}...")
except Exception as e:
    print(f"The following exception was raised while running the experiment {experimentID}: {e}. Check the logs in the experiment directory for more details.")
    raise e
