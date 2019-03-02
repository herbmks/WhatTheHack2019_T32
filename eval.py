import xmltodict
from torch.utils.data import Dataset, DataLoader
from model import WhatTheNet, WTHdataloader, loadmodel, savemodel
import torch

FILE = ['testdata.xbrl']
BATCHSIZE = 2048
OCCUR_THRES = 1000



LOADMODEL = 'model.pth' #'model.pth' # None to disblae

CUDA = False


PREDICT_KEYS = ['GrossOperatingMargin','Assets']


def main():
    assert len(FILE) == 1
    allxml, allkeys = loadpickle()
    evalxml, evalkeys = loadfromfiles(FILE,root='')
    
    cnt = 0
    for k in evalkeys:
        if k in allkeys:
            cnt +=1

    
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
    print('LEN EVALXML', len(evalxml))

    

    for predict_key in PREDICT_KEYS:
        global LOADMODEL
        LOADMODEL = predict_key+'.pth'
        allouts, alllabels, avglabel, avgval = eval_nn(allxml, list(filteredkeys), predict_key)
        # print(allkeys)

        if predict_key not in evalkeys:
            print(predict_key, ' NOT IN EVALKEYS')
        else:
            allouts, alllabels, avglabel, avgval = eval_nn(evalxml, list(filteredkeys), predict_key, avglabel, avgval)
            




def eval_nn(allxml, allkeys, predict_key, avglabel=None, avgval=None):

    # assert len(predict_key) == 1

    print('len allkeys', len(allkeys))
    # for k in predict_key:
    allkeys.remove(predict_key)
    if 'date' in allkeys:
        allkeys.remove('date')
    print('len allkeys2', len(allkeys))

    ds = WTHdataloader(allxml, allkeys, predict_key, avglabel, avgval)
    dl = DataLoader(ds, batch_size=BATCHSIZE, num_workers=4,shuffle=True)

    model = WhatTheNet(len(allkeys), 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=0.0001, params=model.parameters())

    epoch = 0
    if LOADMODEL is not None:
        epoch, model, optimizer, loss, predict_key2 = loadmodel(epoch, model, optimizer, LOADMODEL)
    print(predict_key2, predict_key)
    assert predict_key2 == predict_key

    # bn1 = torch.nn.BatchNorm1d(1).cuda()
    allouts = []
    alllabels = []
    with torch.no_grad():
        for inputs, labels in dl:
            inputs, labels = inputs, labels
            optimizer.zero_grad()
            outputs = model(inputs)
            allouts.append(outputs.detach().cpu())
            alllabels.append(labels.cpu())
            # print('woeifjw',inputs.shape)
            # print('labelsshape', labels.shape)
            # print('outputs', outputs)
            # print(len(labels))
            # labels = bn1(labels)
            # print('labels',labels)
            # print('outputs', outputs.shape)
            for kk in range(15):
                print(labels[kk].item(), outputs[kk].item())
    allouts = torch.cat(allouts)
    alllabels = torch.cat(alllabels)
    avglabel = ds.avglabel
    avgval = ds.avgval
    allouts *= avglabel
    alllabels *= avglabel


    return allouts, alllabels, ds.avglabel, ds.avgval
        





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

if __name__ == "__main__":
    main()