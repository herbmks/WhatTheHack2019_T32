import os
import pickle
import xmltodict
import copy
import numpy as np
import torch
import utils

# SETTINGS

CACHED = True

OCCUR_THRES = 3000
BATCHSIZE = 256*8
LR = 0.0005


LOADMODEL = None
SAVEMODEL = None

MAX_EPOCHS = 80

# LOADMODEL = 'model.pth' #'model.pth' # None to disblae
SAVEMODEL = 'model.pth' # None to disable
CUDA = True


# PREDICT_KEYS = ['AccruedChargesDeferredIncome', 'AmountsPayable', 'ProvisionsDeferredTaxes', 'EquityLiabilities', 'Equity', 'Capital', 'Reserves', 'AccumulatedProfitsLosses', 'InvestmentGrants']
# PREDICT_KEYS = ['AccruedChargesDeferredIncome', 'AmountsPayable', 'ProvisionsDeferredTaxes', 'EquityLiabilities', 'Equity', 'Capital', 'Reserves', 'AccumulatedProfitsLosses', 'InvestmentGrants','GainLossOrdinaryActivitiesBeforeTaxes']
# PREDICT_KEYS = ['AmountsPayable']
# PREDICT_KEYS = ['Equity']

# PREDICT_KEYS = ['AccruedChargesDeferredIncome','GainLossOrdinaryActivitiesBeforeTaxes','FinancialFixedAssetsAcquisitionValue','FinancialFixedAssets','IncomeTaxe','NumberEmployeesPersonnelRegisterClosingDateFinancialYearFullTime','RemunerationDirectSocialBenefits','Assets','AmountsPayableMoreOneYearMoreOneNotMoreFiveYears','AmountsPayableMoreOneYearMoreOneNotMoreFiveYears','RemunerationSocialSecurityPensions','TransfersToCapitalReserves','EmployeesRecordedPersonnelRegisterTotalNumberClosingDate','AmountsPayableMoreOneYearCreditInstitutionsLeasingOtherSimilarObligations','NumberEmployeesPersonnelRegisterClosingDateFinancialYearContractIndefinitePeriodTotalFullTimeEquivalents','NumberEmployeesPersonnelRegisterClosingDateFinancialYearMenTotalFullTimeEquivalents','PersonnelCostsTotal','CurrentPortionAmountsPayableMoreOneYearFallingDueWithinOneYear','GrossOperatingMargin','Capital','LegalReserve','FurnitureVehicles','GainLossPeriod','AmountsPayable','GainLossToBeAppropriated','OperatingProfitLoss','IntangibleFixedAssetsDepreciationsAmountWrittenDown','Taxes','IntangibleFixedAssetsDepreciationsAmountWrittenDown','TangibleFixedAssetsAcquisitionIncludingProducedFixedAssets','GainLossBeforeTaxes','OtherAmountsPayableWithinOneYear','TradeDebtsPayableWithinOneYear','CurrentsAssets','Equity','FinancialDebtsRemainingTermMoreOneYear','AmountsPayableWithinOneYear','FinancialIncome','AccumulatedProfitsLosses']

PREDICT_KEYS = ['GrossOperatingMargin']
def main():


    if CACHED:
        allxml, allkeys = utils.loadpickle()
    else:
        allfiles = os.listdir('data')
        allxml, allkeys = utils.loadfromfiles(allfiles)


    
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

    predict_key = most_common_key

    for predict_key in PREDICT_KEYS:
        # print('PREDICTING KEY',  predict_key)
        # if predict_key not in list(filteredkeys):
        #     print('this key is not in features!')
        #     continue

        # predict_key = 'GrossOperatingMargin' # PREDICT KEY
        global SAVEMODEL
        SAVEMODEL = predict_key+'.pth'
        train_nn(allxml, list(filteredkeys), predict_key)

def train_nn(allxml, allkeys, predict_key):
    from torch.utils.data import Dataset, DataLoader
    from model import WhatTheNet, WTHdataloader, loadmodel, savemodel
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

    ds = WTHdataloader(allxml, allkeys, predict_key)
    if len(ds) == 0:
        return
    dl = DataLoader(ds, batch_size=BATCHSIZE, num_workers=4,shuffle=True)

    model = WhatTheNet(len(allkeys), 1).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=LR, params=model.parameters())


    epoch = 0

    if LOADMODEL is not None:
        epoch, model, optimizer, loss = loadmodel(epoch, model, optimizer, LOADMODEL)

    # bn1 = torch.nn.BatchNorm1d(1).cuda()
    while(True):
        print('Epoch ', epoch)
        if epoch > MAX_EPOCHS:
            break
        for inputs, labels, idx in dl:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('woeifjw',inputs.shape)
            # print('labelsshape', labels.shape)
            # print('outputs', outputs)
            # print(len(labels))
            # labels = bn1(labels)
            # print('labels',labels)
            # print('outputs', outputs.shape)
            for kk in range(15):
                print(labels[kk].item(), outputs[kk].item())
            loss = criterion(outputs, labels)
            loss.backward()
            print('loss', loss)
            optimizer.step()
        epoch += 1
        if SAVEMODEL is not None:
            savemodel(epoch, model, optimizer, loss, predict_key, SAVEMODEL)

        
        # print(trainvec)

if __name__ == "__main__":
    main()
# except:
#     pass
# finally:
#     import IPython
#     IPython.embed()
    