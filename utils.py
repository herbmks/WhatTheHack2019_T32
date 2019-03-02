import os
import numpy as np
import xmltodict


def find_amount_of_common_keys(allxml, allkeys):
    allkeyscopy = {} #et(copy.deepcopy(allkeys))
    for k in allkeys:
        allkeyscopy[k] = 0
    for x in allxml:
        thiskeys = x.keys()
        for k in thiskeys:
            if k in allkeyscopy:
                allkeyscopy[k] += 1
            else:
                allkeyscopy[k] = 1
        
    cntalways = 0
    nbfiles = len(allxml)
    for k, val in allkeyscopy.items():
        if val == nbfiles:
            cntalways += 1
    return cntalways, allkeyscopy


def loadfromfiles(allfiles, root = ''):
    allkeys = set()
    allxml = []
    for fn in allfiles:
        # print(fn)
        with open(root+fn) as fd:
            doc = xmltodict.parse(fd.read())
            a = doc['xbrli:xbrl']
            relevantkeys = {}
            for key, val in a.items():
                if 'pfs:' in key:
                    key = key.replace('pfs:', '')
                    if not isinstance(val, list): # multiple values
                        val = [val]
                    for v in val:
                        # print(v)
                        # if '@xmlns:pfs':
                        #     datefield = (v['@xmlns:pfs'])
                        #     date = datefield.split('/')[-1]
                        #     relevantkeys['date'] = date


                        if '#text' in v:
                            try:
                                v = float(v['#text'])
                                allkeys.add(key)
                                if key not in relevantkeys:
                                    relevantkeys[key] = [v]
                                else:
                                    relevantkeys[key].append(v)
                            except:
                                print('whoops')
            allxml.append(relevantkeys)
    return allxml, allkeys

def loadpickle():
    import pickle
    with open('allxml.p', 'rb') as fp:
        allxml = pickle.load(fp)
    with open('allkeys.p', 'rb') as fp:
        allkeys = pickle.load(fp)
    return allxml, allkeys
