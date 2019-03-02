# from xbrl import XBRLParser, GAAP, GAAPSerializer
# import xbrl as xbrl2
# xbrl_parser = XBRLParser()

import xmltodict
import os
import pickle


allfiles = os.listdir('data')

##  This file reads all the XML stuff and 
##  drops it in a list called allxml
## each element of allxml is a dictionary
## where the keys are the XBLR fields and the dictionary
## values are the numbers from the XBLR files
## when the same XBLR field is multiple times in a file, 
## the dictionary value is a list [value1, value2]
## the dictionary is 


try:
    # loop over all files
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
                        # else:
                            # print('blub',v)

            # print(relevantkeys)

    print(allkeys)

    with open('allxml.p', 'wb') as fp:
        pickle.dump(allxml, fp,protocol=2)
    with open('allkeys.p', 'wb') as fp:
        pickle.dump(allkeys, fp,protocol=2)
except:
    pass
finally:
    import IPython
    IPython.embed()

        # a = a['pfs:FurnitureVehicles']
        # a = a[1]
        # a = a['#text']
        # print(a)






















# fn = "data/lnsmljmw.xbrl"

# # # with open(fn, 'r') as f:
# # #     data = f.read()
# # #     print(data)

# # # xbrl = xbrl_parser.parse(open(fn))
# # # gaap_obj = xbrl_parser.parseGAAP(xbrl, doc_date="20131228", context="current", ignore_errors=0)

# # import xml.etree.ElementTree as ET

# # tree = ET.parse(fn)
# # root = tree.getroot()
# # print(root)


# # def print_subtree(subtree):
# #     for y in subtree:
# #         print("\t", y.tag, ":", y.text)

# # for x in root:
# #     print(x.tag, x.attrib)
# #     print_subtree(x.getchildren())

# import untangle
# import xmltodict
# # obj = untangle.parse(fn)
# # print(obj)



# with open(fn) as fd:
#     doc = xmltodict.parse(fd.read())
#     a = doc['xbrli:xbrl']
#     a = a['pfs:FurnitureVehicles']
#     a = a[1]
#     a = a['#text']
#     print(a)