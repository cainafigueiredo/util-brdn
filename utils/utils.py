import re
import json
import hashlib

def readContentFromFile(filepath):
    with open(filepath) as f:
        return f.read()

def cleanPreds(preds):
    '''Clean +, -, and # from modes'''
    ret = set()
    for line in preds:
        m = re.search('^(\w+)\(([\w, +\-\#\`]+)*\).$', line)
        if m:
            relation = m.group(1)
            relation = re.sub('[+\-\#\` ]', '', relation)
            entities = m.group(2)
            entities = re.sub('[+\-\#\` ]', '', entities)
            ret.add(relation + '(' + entities + ').')
    return list(ret)

def extractRelationNameFromPred(pred):
    '''Get relation name from a predicate.'''
    relation = re.findall(r"^(\w+)\(.*\)\.$", pred)
    try: 
        return relation[0]
    except:
        return None
    
def getHashFromDict(dictData: dict):
    settings = json.dumps(dictData, sort_keys = True)
    sha256_hash = hashlib.sha256()
    sha256_hash.update(settings.encode('utf-8'))
    hash_result = sha256_hash.hexdigest()
    return hash_result