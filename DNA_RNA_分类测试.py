from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from scipy import stats
from scipy.stats import ttest_rel
import pandas
import numpy as np

import os
import xlrd
import xlwt
import re
import numpy
import math
# from xlutils.copy import copy

from sklearn.model_selection import ShuffleSplit
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import xlrd
from  sklearn import  cross_validation
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import xlrd
import numpy as np
from  sklearn import  cross_validation
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

#
#
# data = xlrd.open_workbook(r'E:\2018.1.10\特征表\总表_feature.xlsx')
# table = data.sheets()[0]
# nrows = table.nrows-1
# ncols = table.ncols-1
#
# # print(nrows)
# # print(ncols-1)
# matrix = np.zeros((nrows,ncols))
#
# for i in range(nrows):
#     for j in range(ncols):
#             matrix[i,j] = float(table.cell_value(i+1,j+1))

# DNA_dataset = np.array(matrix[:369,:])
# RNA_dataset = np.array(matrix[369:,:])

# print(DNA_dataset.shape)
# print(RNA_dataset.shape)

# print(DNA_dataset)
# print(RNA_dataset)
# head_name = {}
# for i in range(686):
#     head_name[i] = 0

# for j in range(ncols):
#     if j:
#         head_name[table.cell_value(0, j)] = 0

# label = np.arange(348)
#
# for i in range(348):
#     if i <174:
#         label[i] = 1
#     else:
#         label[i] = -1


import h5py
import os


data = np.zeros((1,5,26))
label = np.zeros((1))

#RNA结合蛋白
print("please input the hdf5 path：")
hdf5_Path = r'F:\DNA_RNA_deeplearning\RNA_file\RNA_bindingsite_5x25'
os.chdir(hdf5_Path)
print("you input hdf5 path is :", os.getcwd())
hdf5_Path_listdir = os.listdir(hdf5_Path)

row_num = 0
for hdf5_file_name in hdf5_Path_listdir:

    f = h5py.File(hdf5_file_name, 'r')
    data_x = f['train_x']
    data_y = f['train_y']

    data = np.append(data,data_x, axis = 0)
    label = np.append(label,data_y, axis = 0)
    row_num = row_num + data_x.shape[0]
        # print(train.shape[0])

label[:] = -1

#DNA结合蛋白
print("please input the hdf5 path：")
hdf5_Path = r'F:\DNA_RNA_deeplearning\DNA_file\DNA_binding_site_2x25'
os.chdir(hdf5_Path)
print("you input hdf5 path is :", os.getcwd())
hdf5_Path_listdir = os.listdir(hdf5_Path)

row_num = 0
for hdf5_file_name in hdf5_Path_listdir:

    f = h5py.File(hdf5_file_name, 'r')
    data_x = f['train_x']
    data_y = f['train_y']

    data = np.append(data,data_x, axis = 0)
    label = np.append(label,data_y, axis = 0)
    row_num = row_num + data_x.shape[0]
        # print(train.shape[0])




data = data[1:,:,:]
label = label[1:]

print(data.shape)


reshaped_xs = np.reshape(data, (
    data.shape[0],
    130,))


print(reshaped_xs.shape)




result = 0
loop_num  = 1
fpr_sum = 0
tpr_sum = 0
counter = 0

from sklearn.neural_network import MLPClassifier

for i in range(2):
    counter =counter + 1
    print(counter)
    # X_train, X_test = train_test_split(DNA_dataset,test_size=0.527)
    #
    # preparation_dataset = np.concatenate((X_train,RNA_dataset),axis=0)



    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(reshaped_xs, label,test_size=0.30)
    # X_train = preprocessing.scale(X_train)
    # X_test = preprocessing.scale(X_test)

    from sklearn.linear_model import RidgeClassifier
    import sklearn.linear_model as linear_model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.grid_search import GridSearchCV


    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': list(range(1,1000,500))},
    #                     {'kernel': ['linear'], 'C': list(range(1,1000,500))}]

    # tuned_parameters = [{'kernel': ['rbf'], 'C': [1,10,100,1000]},
    #                     {'kernel': ['linear'], 'C': [1,10,100,1000]}]
    #
    # print('开始训练')
    # gsearch1 = GridSearchCV(SVC(probability=True),param_grid = tuned_parameters,scoring = 'precision',cv = 5)
    # print('输入训练数据')
    # gsearch1.fit(X_train,y_train)
    # print("训练结束")
    # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)



    # clf = gsearch1
    #clf = SVC(probability=True)
    #clf = RandomForestClassifier()
    #clf = RandomForestClassifier()
    #clf = LogisticRegression()
    clf.fit(X_train,y_train)



    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    probas_ = clf.predict_proba(X_test)

    y_pred = clf.predict(X_test)

    result = result + accuracy_score(y_test, y_pred)
    print("accuracy:", accuracy_score(y_test, y_pred))
    from sklearn.metrics import precision_score
    print("precision:",precision_score(y_test, y_pred))
    from sklearn.metrics import recall_score
    print("recall:",recall_score(y_test, y_pred))
    from sklearn.metrics import f1_score
    print("f1:",f1_score(y_test, y_pred))
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))

    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    print("roc_auc:",roc_auc)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=1,label='AUC = %0.2f' % (roc_auc))


from sklearn.metrics import classification_report
target_names = ['DNA', 'RNA']

# print("accuracy:",result/loop_num)
# 绘图功能
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',label='Luck', alpha=1)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = roc_auc

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#绘图功能结束

#
# #独立测试
# data_i = xlrd.open_workbook(r'E:\2018.1.10\独立测试集\feture.xlsx')
# table_i = data_i.sheets()[0]
# nrows = table_i.nrows-1
# ncols = table_i.ncols-1
#
# # print(nrows)
# # print(ncols-1)
# matrix = np.zeros((nrows,ncols))
#
# for i in range(nrows):
#     for j in range(ncols):
#             matrix[i,j] = float(table_i.cell_value(i+1,j+1))
#
# DNA_dataset_i = np.array(matrix[:88,:])
# RNA_dataset_i = np.array(matrix[88:,:])
#
# print(DNA_dataset_i)
# print(RNA_dataset_i)
#
# label_i = np.arange(176)
#
# for i in range(176):
#     if i <88:
#         label_i[i] = 1
#     else:
#         label_i[i] = -1
#
# X_train, X_test, y_train, y_test = train_test_split(matrix, label_i, random_state= 33,test_size=0)
# X_train = preprocessing.scale(X_train)
#
#
#
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
#
# probas_ = clf.predict_proba(X_train)
#
# y_pred = clf.predict(X_train)
#
# result = result + accuracy_score(y_train, y_pred)
# print("accuracy:", accuracy_score(y_train, y_pred))
# from sklearn.metrics import precision_score
# print("precision:",precision_score(y_train, y_pred))
# from sklearn.metrics import recall_score
# print("recall:",recall_score(y_train, y_pred))
# from sklearn.metrics import f1_score
# print("f1:",f1_score(y_train, y_pred))
# # Compute ROC curve and area the curve
# fpr, tpr, thresholds = roc_curve(y_train, probas_[:, 1])
# tprs.append(interp(mean_fpr, fpr, tpr))
#
# tprs[-1][0] = 0.0
# roc_auc = auc(fpr, tpr)
# print("roc_auc:",roc_auc)
# aucs.append(roc_auc)
# plt.plot(fpr, tpr, lw=2, alpha=1,label='AUC = %0.2f' % (roc_auc))
#
#
# from sklearn.metrics import classification_report
# target_names = ['DNA', 'RNA']
#
# # print("accuracy:",result/loop_num)
# # 绘图功能
# plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',label='Luck', alpha=1)
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = roc_auc
#
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
# # 绘图功能结束



#
# print(result/5)


    # X_new = model.transform(preparation_dataset)

# print(head_name)
#
# dict= sorted(head_name.items(), key=lambda d:d[1], reverse = True)
#
#
# for temp in dict:


