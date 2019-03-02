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


# PREDICT_KEYS = ['GrossOperatingMargin', 'Assets']
# PREDICT_KEYS = ['AccruedChargesDeferredIncome', 'AmountsPayable', 'ProvisionsDeferredTaxes', 'EquityLiabilities', 'Equity', 'Capital', 'Reserves', 'AccumulatedProfitsLosses', 'InvestmentGrants']
# PREDICT_KEYS = ['AccruedChargesDeferredIncome', 'AmountsPayable', 'ProvisionsDeferredTaxes', 'EquityLiabilities', 'Equity', 'Capital', 'Reserves', 'AccumulatedProfitsLosses', 'InvestmentGrants','GainLossOrdinaryActivitiesBeforeTaxes']
# PREDICT_KEYS = ['AmountsPayable']
# PREDICT_KEYS = ['Equity']



# PREDICT_KEYS = ['AccruedChargesDeferredIncome','GainLossOrdinaryActivitiesBeforeTaxes','FinancialFixedAssetsAcquisitionValue','FinancialFixedAssets','IncomeTaxes','NumberEmployeesPersonnelRegisterClosingDateFinancialYearFullTime','RemunerationDirectSocialBenefits','Assets','AmountsPayableMoreOneYearMoreOneNotMoreFiveYears','AmountsPayableMoreOneYearMoreOneNotMoreFiveYears','RemunerationSocialSecurityPensions','TransfersToCapitalReserves','EmployeesRecordedPersonnelRegisterTotalNumberClosingDate','AmountsPayableMoreOneYearCreditInstitutionsLeasingOtherSimilarObligations','NumberEmployeesPersonnelRegisterClosingDateFinancialYearContractIndefinitePeriodTotalFullTimeEquivalents','NumberEmployeesPersonnelRegisterClosingDateFinancialYearMenTotalFullTimeEquivalents','PersonnelCostsTotal','CurrentPortionAmountsPayableMoreOneYearFallingDueWithinOneYear','GrossOperatingMargin','Capital','LegalReserve','FurnitureVehicles','GainLossPeriod','AmountsPayable','GainLossToBeAppropriated','OperatingProfitLoss','IntangibleFixedAssetsDepreciationsAmountWrittenDown','Taxes','IntangibleFixedAssetsDepreciationsAmountWrittenDown','TangibleFixedAssetsAcquisitionIncludingProducedFixedAssets','GainLossBeforeTaxes','OtherAmountsPayableWithinOneYear','TradeDebtsPayableWithinOneYear','CurrentsAssets','Equity','FinancialDebtsRemainingTermMoreOneYear','AmountsPayableWithinOneYear','FinancialIncome','AccumulatedProfitsLosses']
# PREDICT_KEYS = ['AmountsPayable','TransfersToCapitalReserves']

def removenan(vec):
    ipp = np.invert(np.isnan(vec))
    return vec[ipp]

def main():
    if CACHED:
        allxml, allkeys = utils.loadpickle()
    else:
        allfiles = os.listdir('data')
        allxml, allkeys = utils.loadfromfiles(allfiles)

    allkeys = list(allkeys)
    PREDICT_KEYS = allkeys
    ind = []
    for pk in PREDICT_KEYS:
        ind.append(allkeys.index(pk))

    ww = np.zeros( (len(allxml), len(PREDICT_KEYS)))
    ww[:] = np.nan

    for i,xml in enumerate(allxml):
        for kk,pk in enumerate(PREDICT_KEYS):
            if pk in xml:
                val = xml[pk]
                if isinstance(val, list):
                    val = val[0]
                ww[i, kk] = val

    for i, pk in enumerate(PREDICT_KEYS):
        vals = (ww[:,i])
        np.savetxt("nl/"+pk+".csv", ww[:,i], delimiter=",")
        vals = removenan(ww[:,i])
        np.savetxt("nlf/"+pk+".csv", vals, delimiter=",")
        # with open('nl'+pk+'.p', 'wb') as fp:
        #     pickle.dump(ww[:,i], fp, protocol=2)
        
    # with open('alllabels.p', 'wb') as fp:
    #     pickle.dump(alllabels, fp, protocol=2)









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
    