import os
import sys
import logging

from copy import copy
from glob import glob
from typing import Optional

FILE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(f"{FILE_DIR}/..")
sys.path.append(f"{PROJECT_DIR}/../srlearn")

from srlearn.database import Database
from srlearn.weight import WeightFactory
from srlearn.rdn import RDNBoost, RDNBoostTransferLearning, TreeBoostler

def getLogger(
    name: str, 
    level: int = logging.DEBUG, 
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    logFile: str = None,
    consoleOutput: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        formatter = logging.Formatter(format)
        if consoleOutput:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if logFile:
            file_handler = logging.FileHandler(logFile)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

def loadDatabase(
    path: str = None,
    folds: list = None,
    useRecursion: bool = False,
    targetPredicate: str = None,
    resetTargetPredicate: bool = False, 
    negPosRatio = 1,
    maxFailedNegSamplingRetries: Optional[int] = 50,
    logger: logging.Logger = None
):
    if not logger:
        logger = getLogger("Database loader", level = logging.DEBUG)

    if not folds:
        logger.warn("All available database folds will be loaded. Set `folds` if it is desired to load only a few fold.")
        folds = glob(f"{path}/fold*")

    else: 
        folds = [f"{path}/{os.path.basename(fold)}" for fold in folds]

    totalFolds = len(folds)
    database = None
    for i, foldPath in enumerate(folds, start = 1):
        databaseFold = Database().fromFiles(
            facts = f"{foldPath}/facts.pl",
            pos = f"{foldPath}/pos.pl",
            neg = f"{foldPath}/neg.pl",
            modes = f"{foldPath}/modes.pl",
            useRecursion = useRecursion
        )
        database = databaseFold if not database else database.merge(databaseFold)
        logger.debug(f"{i}/{totalFolds} folds loaded with success.")

    if resetTargetPredicate:
        logger.warn(f"`resetTargetPredicate` is set to True. Target predicate `{targetPredicate}` will be ignored and target relation will be set to `None`. Set `resetTargetPredicate` to False if it is desired to set `targetPredicate` as the target relation.")
        database = database.resetTargetPredicate()
    else: 
        if targetPredicate:
            database = database.setTargetPredicate(
                targetPredicate, 
                useRecursion = useRecursion,
                negPosRatio = negPosRatio,
                maxFailedNegSamplingRetries = maxFailedNegSamplingRetries
            )
        else: 
            defaultTargetPredicate = database.getTargetRelation()
            if defaultTargetPredicate:
                database.neg = database.generateNegativePredicates(
                    negPosRatio = negPosRatio,
                    maxFailedSamplingRetries = maxFailedNegSamplingRetries,
                    keepCurrentNegativeExamples = True
                )
                logger.warn(f"No target predicate was given. It will be set as `{defaultTargetPredicate}` by default. Set `targetPredicate` if it is desired to set another relation as the target relation.")
            else:
                logger.warn(f"No target predicate was given. It will be set as `None` by default. Set `targetPredicate` if it is desired to set any relation as the target relation.")

    return database

# Learning from scratch using the original RDN-Boost
def runSingleExperiment_OriginalRDNBoost(
    experimentPath: str = ".", 
    databaseTrain: Database = None,
    databaseTest: Database = None,
    nEstimators: int = 10,
    nodeSize: int = 2,
    maxTreeDepth: int = 3,
    negPosRatio: int = 2,
    numberOfClauses: int = 8,
    numberOfCycles: int = 100,
    ignoreSTDOUT: bool = True,
    logger: logging.Logger = None
) -> dict:
    assert databaseTrain is not None
    assert databaseTest is not None

    if not logger:
        logger = getLogger("Original RDN-B", level = logging.DEBUG)

    path = os.path.join(experimentPath, "originalRDNBoost")
    logger.info("RUNNING ORIGINAL RDN-B...")
    logger.info(f"Progress will be store at {path}")
    
    model = RDNBoost(
        n_estimators = nEstimators, 
        node_size = nodeSize, 
        max_tree_depth = maxTreeDepth, 
        neg_pos_ratio = negPosRatio,
        number_of_clauses = numberOfClauses,
        number_of_cycles = numberOfCycles,
        path = path
    )

    logger.info(f"Training the model on the training set...")
    model.fit(databaseTrain, ignoreSTDOUT = ignoreSTDOUT)

    logger.info(f"Evaluating the model on the test set...")
    model._run_inference(databaseTest, ignoreSTDOUT = ignoreSTDOUT)

    model._generate_dotimages()
    dotImages = model._dotimages
    metrics = model._prediction_metrics

    result = {"model": copy(model), "treeImages": copy(dotImages), "metrics": copy(metrics)}
    return result

# Learning from scratch using our implementation analogous to original RDN-Boost
def runSingleExperiment_AnalogousToRDNBoost(
    experimentPath: str = ".", 
    databaseTrain: Database = None,
    databaseTest: Database = None,
    nEstimators: int = 10,
    nodeSize: int = 2,
    maxTreeDepth: int = 3,
    negPosRatio: int = 2,
    numberOfClauses: int = 8,
    numberOfCycles: int = 100,
    ignoreSTDOUT: bool = True,
    logger: logging.Logger = None
) -> dict:
    assert databaseTrain is not None
    assert databaseTest is not None

    if not logger:
        logger = getLogger("Analogous to RDN-B", level = logging.DEBUG)

    path = os.path.join(experimentPath, "analogousToOriginalRDNBoost")
    logger.info("RUNNING OUR APPROACH ANALOGOUS TO THE ORIGINAL RDN-B...")
    logger.info(f"Progress will be store at {path}")

    targetDomainTargetRelation = databaseTrain.getTargetRelation()

    emptySourceDatabase = Database()
    emptySourceDatabase.getTargetRelation = lambda: targetDomainTargetRelation
    emptySourceDatabase.modes = databaseTrain.modes

    weightFactory = WeightFactory()
    weightStrategy = weightFactory.getWeightStrategy("scalar", weight = 1)

    database = Database.prepareTransferLearningDatabase(
        emptySourceDatabase, 
        databaseTrain, 
        weightStrategy = weightStrategy
    )

    utilityAlpha = 1

    model = RDNBoostTransferLearning(
        n_estimators = nEstimators,
        node_size = nodeSize,
        max_tree_depth = maxTreeDepth,
        neg_pos_ratio = negPosRatio,
        number_of_clauses = numberOfClauses,
        number_of_cycles = numberOfCycles,
        source_utility_alpha = utilityAlpha,
        target_utility_alpha = utilityAlpha,
        path = path
    )

    logger.info(f"Training the model on the training set...")
    model.fit(database, ignoreSTDOUT = ignoreSTDOUT)

    logger.info(f"Evaluating the model on the test set...")
    model._run_inference(databaseTest, ignoreSTDOUT = ignoreSTDOUT)

    model._generate_dotimages()
    dotImages = model._dotimages
    metrics = model._prediction_metrics

    result = {"model": copy(model), "treeImages": copy(dotImages), "metrics": copy(metrics)}
    return result

# Our Transfer Learning implementation
def runSingleExperiment_TransferLearning(
    experimentPath: str = ".", 
    sourceDatabase: Database = None,
    targetDatabaseTrain: Database = None,
    targetDatabaseTest: Database = None,
    nEstimators: int = 10,
    nodeSize: int = 2,
    maxTreeDepth: int = 3,
    negPosRatio: int = 2,
    numberOfClauses: int = 8,
    numberOfCycles: int = 100,
    ignoreSTDOUT: bool = True,
    useRecursion: bool = False,
    randomSeed: int = 10,
    maxFailedNegSamplingRetries: int = 50,
    weightStrategy: WeightFactory = None,
    sourceUtilityAlpha: float = 1,
    targetUtilityAlpha: float = 1,
    utilityAlphaSetIter: int = 1,
    relationMapping: dict = None,
    termTypeMapping: dict = None,
    logger: logging.Logger = None,
) -> dict:
    assert sourceDatabase is not None
    assert targetDatabaseTrain is not None
    assert targetDatabaseTest is not None
    assert weightStrategy is not None
    assert sourceUtilityAlpha >= 0
    assert targetUtilityAlpha >= 0
    assert relationMapping is not None
    assert termTypeMapping is not None

    if not logger:
        logger = getLogger("Transfer Learning RDN-B", level = logging.DEBUG)

    targetDomainTargetRelation = targetDatabaseTrain.getTargetRelation()

    path = os.path.join(experimentPath, "transferLearning")
    logger.info("RUNNING TRANSFER LEARNING...")
    logger.info(f"Progress will be store at {path}")

    logger.info("Mapping source domain to the target domain...")

    sourceTargetRelation = [k for k,v in relationMapping.items() if v == targetDomainTargetRelation][0]
    
    logger.debug(f"Relation mapping: {relationMapping}")
    logger.debug(f"Term type mapping: {termTypeMapping}")

    mappedSourceDatabase = sourceDatabase.setTargetPredicate(
        sourceTargetRelation, 
        useRecursion = useRecursion,
        negPosRatio = negPosRatio,
        maxFailedNegSamplingRetries = maxFailedNegSamplingRetries
    )
    mappedSourceDatabase = mappedSourceDatabase.applyMapping(relationMapping, termTypeMapping, "source")

    logger.info("Combining source and target databases...")

    database = Database.prepareTransferLearningDatabase(
        mappedSourceDatabase, 
        targetDatabaseTrain, 
        weightStrategy = weightStrategy
    )

    model = RDNBoostTransferLearning(
        n_estimators = nEstimators,
        node_size = nodeSize,
        max_tree_depth = maxTreeDepth,
        neg_pos_ratio = negPosRatio,
        number_of_clauses = numberOfClauses,
        number_of_cycles = numberOfCycles,
        source_utility_alpha = sourceUtilityAlpha,
        target_utility_alpha = targetUtilityAlpha,
        utility_alpha_set_iter = utilityAlphaSetIter,
        path = path
    )

    logger.info(f"Training the model on the training set...")
    model.fit(database, ignoreSTDOUT = ignoreSTDOUT)

    logger.info(f"Evaluating the model on the test set...")
    model._run_inference(targetDatabaseTest, ignoreSTDOUT = ignoreSTDOUT)

    model._generate_dotimages()
    dotImages = model._dotimages
    metrics = model._prediction_metrics

    result = {"model": copy(model), "treeImages": copy(dotImages), "metrics": copy(metrics)}
    return result