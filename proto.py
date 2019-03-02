import pickle
import numpy as np
with open('allouts.p', 'rb') as fp:
    allouts = pickle.load(fp)
with open('alllabels.p', 'rb') as fp:
    alllabels = pickle.load(fp)


VOF1 = 'AccruedChargesDeferredIncome' #GrossOperatingMargin'
VOF2 = 'FinancialFixedAssets'



pk = ['AccruedChargesDeferredIncome', 'GainLossOrdinaryActivitiesBeforeTaxes', 'FinancialFixedAssetsAcquisitionValue', 'FinancialFixedAssets', 'NumberEmployeesPersonnelRegisterClosingDateFinancialYearFullTime', 'RemunerationDirectSocialBenefits', 'Assets', 'AmountsPayableMoreOneYearMoreOneNotMoreFiveYears', 'AmountsPayableMoreOneYearMoreOneNotMoreFiveYears', 'RemunerationSocialSecurityPensions', 'TransfersToCapitalReserves', 'EmployeesRecordedPersonnelRegisterTotalNumberClosingDate', 'AmountsPayableMoreOneYearCreditInstitutionsLeasingOtherSimilarObligations', 'NumberEmployeesPersonnelRegisterClosingDateFinancialYearContractIndefinitePeriodTotalFullTimeEquivalents', 'NumberEmployeesPersonnelRegisterClosingDateFinancialYearMenTotalFullTimeEquivalents', 'PersonnelCostsTotal', 'CurrentPortionAmountsPayableMoreOneYearFallingDueWithinOneYear', 'GrossOperatingMargin', 'Capital', 'LegalReserve', 'FurnitureVehicles', 'GainLossPeriod', 'AmountsPayable', 'GainLossToBeAppropriated', 'OperatingProfitLoss', 'Equity', 'AccumulatedProfitsLosses']
# pk = ['GrossOperatingMargin']
pkind = np.asarray([0 ,1 ,2 ,3 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,34,38])
# pkind = np.asarray([0])

print('allouts',allouts)
print('alllabels',alllabels)

# import random
# alllabels = allouts[:, pkind]* (1+ (random.random()-0.5) *0.01)
allouts = allouts[:, pkind] 
alllabels = alllabels[:, pkind] 
print(allouts.shape[1], len(pk))

m = np.nanmean(allouts, axis=0)
meanz = np.nanmean(abs(allouts), axis=1)
alldiff = (allouts - alllabels)/meanz[:, None]
MSE = alldiff**2
indicatorperformace = np.nanmean(abs(MSE), axis=0) # higher is worse
for p, k in zip(pk, indicatorperformace):
    print(p,k)

print('indicatorperformancee',indicatorperformace)

# goodindictators = indicatorperformace < 0.2
# goodindictatorsind = np.argwhere(goodindictators)
# print('## GOOD INDICATORS')
# for i in goodindictatorsind:
#     print(pk[i], indicatorperformace[i])
# print(alldiff.shape)


# BEST indicators



import IPython




def removenan(vec):
    ipp = np.invert(np.isnan(vec))
    return vec[ipp]


# final loop 0 AccruedChargesDeferredIncome
# final loop 1 GainLossOrdinaryActivitiesBeforeTaxes
# final loop 2 FinancialFixedAssetsAcquisitionValue
# final loop 3 FinancialFixedAssets
# final loop 5 NumberEmployeesPersonnelRegisterClosingDateFinancialYearFullTime
# final loop 6 RemunerationDirectSocialBenefits
# final loop 7 Assets
# final loop 8 AmountsPayableMoreOneYearMoreOneNotMoreFiveYears
# final loop 9 AmountsPayableMoreOneYearMoreOneNotMoreFiveYears
# final loop 10 RemunerationSocialSecurityPensions
# final loop 11 TransfersToCapitalReserves
# final loop 12 EmployeesRecordedPersonnelRegisterTotalNumberClosingDate
# final loop 13 AmountsPayableMoreOneYearCreditInstitutionsLeasingOtherSimilarObligations
# final loop 14 NumberEmployeesPersonnelRegisterClosingDateFinancialYearContractIndefinitePeriodTotalFullTimeEquivalents
# final loop 15 NumberEmployeesPersonnelRegisterClosingDateFinancialYearMenTotalFullTimeEquivalents
# final loop 16 PersonnelCostsTotal
# final loop 17 CurrentPortionAmountsPayableMoreOneYearFallingDueWithinOneYear
# final loop 18 GrossOperatingMargin
# final loop 19 Capital
# final loop 20 LegalReserve
# final loop 21 FurnitureVehicles
# final loop 22 GainLossPeriod
# final loop 23 AmountsPayable
# final loop 24 GainLossToBeAppropriated
# final loop 25 OperatingProfitLoss
# final loop 34 Equity
# final loop 38 AccumulatedProfitsLosses





# IPython.embed()




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









alldiff =  (allouts - alllabels)**2
# alldiff[alldiff < 2] =  np.nan
pp = [VOF1,VOF2]
ppind = []
for p in pp:
    ind = pk.index(p)
    ppind.append(ind)
print(ppind)

import random

vec1 = alldiff[:,ppind[0]]
#vec2 = allouts[:,ppind[0]]#*(random.random()-0.5)**2
vec2 = alldiff[:,ppind[1]]

# goodind = []
# for i in range(len(vec1)):
#     if np.isnan(vec1[i]) or np.isnan(vec2[i]):
#         continue
#     # if vec1[i] > 2:
#     #     continue
#     # if vec2[i] > 2:
#     #     continue
#     goodind.append(i)
# goodind = np.asarray(goodind)
# vec1 = vec1[goodind]
# vec2 = vec2[goodind]

# print('nann', removenan(vec1))
import matplotlib.pyplot as plt
# plt.figure()
# plt.plot((vec1),label='outs')
# plt.plot((vec2),label='labels')
# plt.legend()
# plt.show()




# print('var1', alldiff[:,ppind[0]].shape)
# print('var2', alldiff[:,ppind[1]].shape)


# print(vec1)
# print(vec2)
# goodind = []

# print(goodind)


# vec1 = vec1[goodind]
# vec2 = vec2[goodind]

plt.figure()
plt.title('Clustering')
plt.scatter(vec1, vec2)
plt.xlabel(VOF1)
plt.ylabel(VOF2)
plt.show()























# AccruedChargesDeferredIncome
# NumberEmployeesPersonnelRegisterClosingDateFinancialYearContractIndefinitePeriodTotalFullTimeEquivalents
# NumberEmployeesPersonnelRegisterClosingDateFinancialYearMenTotalFullTimeEquivalents
# LegalReserve
# FurnitureVehicles


# # INDICATOR PERFORMANCE
# AccruedChargesDeferredIncome 0.05652927518619886
# GainLossOrdinaryActivitiesBeforeTaxes nan
# FinancialFixedAssetsAcquisitionValue 3.9247521987150558
# FinancialFixedAssets 5.7378317677758925
# NumberEmployeesPersonnelRegisterClosingDateFinancialYearFullTime 6.748380105326195e-11
# RemunerationDirectSocialBenefits 0.35886107816815643
# Assets 20.064535013375533
# AmountsPayableMoreOneYearMoreOneNotMoreFiveYears 3.9685598599500644
# AmountsPayableMoreOneYearMoreOneNotMoreFiveYears 3.9685598599500644
# RemunerationSocialSecurityPensions 0.4616849707965157
# TransfersToCapitalReserves 0.2534721885922802
# EmployeesRecordedPersonnelRegisterTotalNumberClosingDate 1.7044894593616426e-10
# AmountsPayableMoreOneYearCreditInstitutionsLeasingOtherSimilarObligations 0.5419571352274668
# NumberEmployeesPersonnelRegisterClosingDateFinancialYearContractIndefinitePeriodTotalFullTimeEquivalents 1.0179492725860331e-10
# NumberEmployeesPersonnelRegisterClosingDateFinancialYearMenTotalFullTimeEquivalents 8.920868637221757e-11
# PersonnelCostsTotal 0.8726873213114446
# CurrentPortionAmountsPayableMoreOneYearFallingDueWithinOneYear 1.0379976366663772
# GrossOperatingMargin 0.6117256215235196
# Capital 6.6066480802728105
# LegalReserve 0.010698909251207927
# FurnitureVehicles 0.07559326334206917
# GainLossPeriod 0.6163071130925203
# AmountsPayable 24.82710686840821
# GainLossToBeAppropriated 0.2512007048863756
# OperatingProfitLoss 0.38149864233785424
# Equity 20.14265106404957
# AccumulatedProfitsLosses 5.69591486564097