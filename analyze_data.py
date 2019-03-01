import os
import pickle
import xmltodict
import copy
import numpy as np
import torch

# SETTINGS

CACHED = True

OCCUR_THRES = 5000





def main():

    if CACHED:
        allxml, allkeys = loadpickle()
    else:
        allfiles = os.listdir('data')
        allxml, allkeys = loadfromfiles(allfiles)


    
    amount_of_commonkeys, freq = find_amount_of_common_keys(allxml, allkeys)
    # print(freq)
    filteredkeys = set()
    most_common_key = None
    most_common_key_freq = 0
    for k, v in freq.items():
        # print(k,v)
        if v > OCCUR_THRES:
            filteredkeys.add(k)
        if v > most_common_key_freq:
            most_common_key = k
            most_common_key_freq = v

    # print(len(filteredkeys))
    print('most common keys', most_common_key)

    predict_key = most_common_key
    train_nn(allxml, allkeys, predict_key)

def train_nn(allxml, allkeys, predict_key):
    from torch.utils.data import Dataset, DataLoader
    from model import WhatTheNet, WTHdataloader
    # assert len(predict_key) == 1

    print('len allkeys', len(allkeys))
    # for k in predict_key:
    allkeys.remove(predict_key)
    if 'date' in allkeys:
        allkeys.remove('date')
    print('len allkeys2', len(allkeys))

    ds = WTHdataloader(allxml, allkeys, predict_key)
    dl = DataLoader(ds, batch_size=256)

    model = WhatTheNet(len(allkeys), len(predict_key))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())


    while(True):
        print('Epoch ')
        for inputs, labels in dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('woeifjw',inputs.shape)
            # print('labelsshape', labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            print('loss', loss)
            optimizer.step()

        
        # print(trainvec)






    


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


def loadpickle():
    import pickle
    with open('allxml.p', 'rb') as fp:
        allxml = pickle.load(fp)
    with open('allkeys.p', 'rb') as fp:
        allkeys = pickle.load(fp)
    return allxml, allkeys


def loadfromfiles(allfiles):
    allkeys = set()
    allxml = []
    for fn in allfiles:
        # print(fn)
        with open('data/'+fn) as fd:
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
                        if '@xmlns:pfs':
                            datefield = (v['@xmlns:pfs'])
                            date = datefield.split('/')[-1]
                            relevantkeys['date'] = date


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


if __name__ == "__main__":
    main()
# except:
#     pass
# finally:
#     import IPython
#     IPython.embed()
    