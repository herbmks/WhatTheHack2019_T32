import xmltodict
from torch.utils.data import Dataset, DataLoader
from model import WhatTheNet, WTHdataloader, loadmodel, savemodel
import torch
import utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 
FILE = ['testdata.xbrl']
DOFILE = False
BATCHSIZE = 2048
OCCUR_THRES = 3000

DOPLOT = True

LOADMODEL = 'model.pth' #'model.pth' # None to disblae

CUDA = False

ONLY_ALL_PREDICT_KEYS = False




# PREDICT_KEYS = ['GrossOperatingMargin']
PREDICT_KEYS = ['AccruedChargesDeferredIncome','GainLossOrdinaryActivitiesBeforeTaxes','FinancialFixedAssetsAcquisitionValue','FinancialFixedAssets','IncomeTaxe','NumberEmployeesPersonnelRegisterClosingDateFinancialYearFullTime','RemunerationDirectSocialBenefits','Assets','AmountsPayableMoreOneYearMoreOneNotMoreFiveYears','AmountsPayableMoreOneYearMoreOneNotMoreFiveYears','RemunerationSocialSecurityPensions','TransfersToCapitalReserves','EmployeesRecordedPersonnelRegisterTotalNumberClosingDate','AmountsPayableMoreOneYearCreditInstitutionsLeasingOtherSimilarObligations','NumberEmployeesPersonnelRegisterClosingDateFinancialYearContractIndefinitePeriodTotalFullTimeEquivalents','NumberEmployeesPersonnelRegisterClosingDateFinancialYearMenTotalFullTimeEquivalents','PersonnelCostsTotal','CurrentPortionAmountsPayableMoreOneYearFallingDueWithinOneYear','GrossOperatingMargin','Capital','LegalReserve','FurnitureVehicles','GainLossPeriod','AmountsPayable','GainLossToBeAppropriated','OperatingProfitLoss','IntangibleFixedAssetsDepreciationsAmountWrittenDown','Taxes','IntangibleFixedAssetsDepreciationsAmountWrittenDown','TangibleFixedAssetsAcquisitionIncludingProducedFixedAssets','GainLossBeforeTaxes','OtherAmountsPayableWithinOneYear','TradeDebtsPayableWithinOneYear','CurrentsAssets','Equity','FinancialDebtsRemainingTermMoreOneYear','AmountsPayableWithinOneYear','FinancialIncome','AccumulatedProfitsLosses']

# PREDICT_KEYS = ['GrossOperatingMargin']
# PREDICT_KEYS = ['Equity']


def main():
    assert len(FILE) == 1
    allxml, allkeys = utils.loadpickle()
    if DOFILE:
        evalxml, evalkeys = utils.loadfromfiles(FILE,root='') 
        cnt = 0
        for k in evalkeys:
            if k in allkeys:
                cnt +=1
        print('LEN EVALXML', len(evalxml))


    
    amount_of_commonkeys, freq = utils.find_amount_of_common_keys(allxml, allkeys)
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


    if ONLY_ALL_PREDICT_KEYS:
        # Only evaluate items that have all PREDICT_KEYS
        filteredxml = []
        for xml in allxml:
            allin = True
            for predict_key in PREDICT_KEYS:
                if predict_key not in xml:
                    allin=False
            if allin:
                filteredxml.append(xml)
        allxml = filteredxml

    if DOFILE:
        ## Check if all eval set have all predict keys
        for xml in evalxml:
            for predict_key in PREDICT_KEYS:
                if predict_key not in xml:
                    raise ValueError('PREDICTKEY NOT IN  DATA')




    
    outs = {}
    labels = {}
    diffs = {}
    ids = {}
    for predict_key in PREDICT_KEYS:
        try:
            if not os.path.isfile('models/'+predict_key+'.pth'):
                continue

            global LOADMODEL
            LOADMODEL = predict_key+'.pth'
            allouts, alllabels, avglabel, avgval, allids = eval_nn(allxml, list(filteredkeys), predict_key)
            if allouts is None:
                continue
            print(allouts)
            outs[predict_key] = allouts.squeeze(1)
            print('ALLOUTS SHAPE', allouts.shape)
            labels[predict_key] = alllabels.squeeze(1)
            diffs[predict_key] = allouts-alllabels

            print('allids SHAPE', allids.shape)
            ids[predict_key] = allids.squeeze(1)
            # print(allkeys)
            if DOFILE:
                if predict_key not in evalkeys:
                    print(predict_key, ' NOT IN EVALKEYS')
                else:
                    allouts, alllabels, avglabel, avgval = eval_nn(evalxml, list(filteredkeys), predict_key, avglabel, avgval)
        except:
            pass
    # remap
    allouts = np.ones((len(allxml), len(PREDICT_KEYS)))*-10
    alllabels = np.ones((len(allxml), len(PREDICT_KEYS)))*-10
    allouts[:] = np.nan
    alllabels[:] = np.nan

    for pk, predict_key in enumerate(PREDICT_KEYS):
        if not os.path.isfile('models/'+predict_key+'.pth'):
            continue
        print('final loop',pk, predict_key)
        nbel = len(outs[predict_key])
        thiso = outs[predict_key]
        thisl = labels[predict_key]
        thisi = ids[predict_key]
        # for i in range(nbel):
        ii = thisi.numpy().astype(np.int)
        #     # print('ii', ii)
        #     allouts[ii, pk] = thiso[i].item()
        #     alllabels[ ii, pk] = thisl[i].item()
        # ii = thisi[i]
        allouts[ii, pk] = thiso.numpy()
        alllabels[ii, pk] = thisl.numpy()
    
    with open('allouts.p', 'wb') as fp:
        pickle.dump(allouts, fp, protocol=2)
    with open('alllabels.p', 'wb') as fp:
        pickle.dump(alllabels, fp, protocol=2)

    meanz = np.mean(abs(allouts), axis=1)
    print('getting diff')
    alldiff = (allouts - alllabels)/meanz[:, None]
    MSE = alldiff**2
    print('MSEshape', MSE.shape)
    indicatorperformace = np.nanmean(abs(MSE), axis=0) # higher is worse
    print('indicatorperformancee',indicatorperformace)

    goodindictators = indicatorperformace < 0.2
    goodindictatorsind = np.argwhere(goodindictators)
    print('## GOOD INDICATORS')
    for i in goodindictatorsind:
        print(PREDICT_KEYS[i])

    MSE = MSE[:,goodindictators]

    MSE_avg = np.nanmean(abs(MSE), axis=1) # avg mse per statement
    print(MSE_avg)

    # MSE_per_samples = []



    if DOPLOT:
        print('var1', diffs[PREDICT_KEYS[0]].shape)
        print('var2', diffs[PREDICT_KEYS[1]].shape)
        plt.figure()
        plt.scatter(diffs[PREDICT_KEYS[0]], diffs[PREDICT_KEYS[1]])
        plt.show()




def eval_nn(allxml, allkeys, predict_key, avglabel=None, avgval=None):

    # assert len(predict_key) == 1

    print('len allkeys', len(allkeys))
    # for k in predict_key:
    try:
        allkeys.remove(predict_key)
    except:
        print('warning: was not in keys')
    if 'date' in allkeys:
        allkeys.remove('date')
    print('len allkeys2', len(allkeys))

    ds = WTHdataloader(allxml, allkeys, predict_key, avglabel, avgval)
    if len(ds) == 0:
        return None, None, None, None, None
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
    allids= []
    with torch.no_grad():
        for inputs, labels, idx in dl:
            inputs, labels = inputs, labels
            # optimizer.zero_grad()
            outputs = model(inputs)
            allouts.append(outputs.detach().cpu())
            alllabels.append(labels.cpu())
            allids.append(idx.cpu())
            # print('woeifjw',inputs.shape)
            # print('labelsshape', labels.shape)
            # print('outputs', outputs)
            # print(len(labels))
            # labels = bn1(labels)4

            # print('labels',labels.shape)
            # print('outputs', outputs.shape)
            assert labels.shape == outputs.shape
            for kk in range(min(15, len(labels))):
                print(labels[kk].item(), outputs[kk].item())
    allouts = torch.cat(allouts)
    alllabels = torch.cat(alllabels)
    allids = torch.cat(allids)
    avglabel = ds.avglabel
    avgval = ds.avgval
    allouts *= avglabel
    alllabels *= avglabel


    return allouts, alllabels, ds.avglabel, ds.avgval, allids
        
if __name__ == "__main__":
    main()