import scipy.io as sio
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# For Windows.;.;

label_path = 'D:\\imacros'
label_list = os.listdir(label_path)
label_list = [f for f in label_list if f.endswith('.csv')]
phone_list = pd.read_csv('D:\\known_phone_list.csv')

# Label Processing
datepeople_df = pd.DataFrame()
for i, label_file in enumerate(label_list):
    match_dt = re.match("Extract_(?P<day>\d\d)(?P<month>\d\d)(?P<year>\d\d)_(?P<hour>\d\d)(?P<minute>\d\d)(?P<second>\d\d)\.csv",label_file)
    datetimes = match_dt.groupdict()
    datetimes = dict((k, int(v)) for k, v in datetimes.items())
    datetimes['year'] += 2000
    datetimes_ser = pd.Series(datetimes)

    read_df = pd.read_csv(label_path+'\\'+label_file,header=None)
    read_df = read_df[~read_df[0].str.contains('--')]
    people_ser = pd.Series()
    #ff = re.match(phone_list.loc[3].values[0],f.loc[9].values[0])
    for j in range(len(phone_list)):
        mac_addr = phone_list.loc[j].mac
        mac_name = phone_list.loc[j].owner
        people_ser[mac_name] = len(read_df[read_df[0].str.contains(mac_addr)])
    people_ser['total_people'] = sum(people_ser)
    datepeople_ser = pd.concat((datetimes_ser,people_ser))
    datepeople_df = datepeople_df.append(datepeople_ser,ignore_index=True)
datepeople_df = datepeople_df.astype('int')
col_sequence = ['year','month','day','hour','minute','second','total_people','HERO','Zaynab','SI','JY','JS','SC']
datepeople_df = datepeople_df.reindex(columns=col_sequence)

selected_label = datepeople_df[(datepeople_df.day==20)|(datepeople_df.day==21)|(datepeople_df.day==27)|(datepeople_df.day==28)]
csi_list = ['csi'+str(selected_label.iloc[i].year)+str(selected_label.iloc[i].month).zfill(2)+str(selected_label.iloc[i].day)+str(selected_label.iloc[i].hour).zfill(2)+'03' for i in range(len(selected_label))]

selected_label = selected_label.set_index([csi_list])
label = selected_label.total_people.groupby(label.index).mean()
csi_list=list(set(csi_list))

# CSI Processing

print("CSI:Abs value")

#dateid = 'csi201809112003'
csi_summ = pd.DataFrame()
for dateid in csi_list:
    file_path = 'D:\\CSI\\' + dateid
    file_list = os.listdir(file_path)
    file_list.sort()
    file_list.sort(key=len)
    list_10th = [file_list[i] for i in (np.arange(0,1,0.1) * len(file_list)).astype('int')]

    df_sc = pd.DataFrame()
    dict_csi = {}
    scalar_colname = ['timestamp_low','bfee_count','Nrx','Ntx','rssi_a','rssi_b','rssi_c','noise','agc','rate']
    for i, file in enumerate(file_list):
        data_read = sio.loadmat(os.path.join(file_path,file))
        num_search = re.search('\d+',file)
        csi_num = int(num_search.group(0))

        # scalar data
        read_sc0_8 = [data_read['csi_entry'][0][0][a][0][0] for a in range(9)]
        data_sc = pd.DataFrame(read_sc0_8).astype('int').T
        data_sc.columns=scalar_colname[0:9]
        data_sc['perm'] = pd.Series([data_read['csi_entry'][0][0][9][0]])
        data_sc['rate'] = data_read['csi_entry'][0][0][10][0][0]
        data_sc.index = [csi_num]

        # csi info
        csi_raw = data_read['csi_entry'][0][0][11]
        csi_scaled = data_read['csi_entry'][0][0][12]

        # aggregate
        df_sc = df_sc.append(data_sc)
        dict_csi[csi_num] = csi_raw,csi_scaled

        #if file in list_10th:
        #    print(int(100*(i+1)/len(file_list)))

    ntx = int(np.unique(data_sc.Ntx.values))
    nrx = int(np.unique(data_sc.Nrx.values))
    # plot2 : by subcarrier
    print("CSI:to array "+dateid)
    list_abs2 = []
    list_ph2 = []
    for i in range(len(dict_csi)):
        abs_val = np.reshape(np.abs(dict_csi[i+1][1]),((1,90)))
        abs_ph = np.reshape(np.angle(dict_csi[i+1][1]),((1,90)))
        #abs_val = np.mean(np.abs(np.angle(dict_csi[i+1][1])),axis=2)
        if abs_val.shape==(ntx,nrx*30):
            list_abs2.append(abs_val)
            list_ph2.append(abs_ph)
    arr_abs2 = np.stack(list_abs2, axis=-1)
    arr_ph2 = np.stack(list_ph2, axis=-1)

    abs_mean = pd.DataFrame(np.mean(arr_abs2,axis=2),index=[dateid])
    abs_std = pd.DataFrame(np.std(arr_abs2,axis=2),index=[dateid])
    ph_mean = pd.DataFrame(np.mean(arr_ph2,axis=2),index=[dateid])
    ph_std = pd.DataFrame(np.std(arr_ph2,axis=2),index=[dateid])

    name_num = list(map(str,range(1,91)))
    abs_mean.columns = ['abs_mean'+ i for i in name_num]
    abs_std.columns = ['abs_std'+ i for i in name_num]
    ph_mean.columns = ['ph_mean'+ i for i in name_num]
    ph_std.columns = ['ph_std'+ i for i in name_num]

    csi_summ1 = pd.concat((abs_mean,abs_std,ph_mean,ph_std),axis=1)

    csi_summ = csi_summ.append(csi_summ1)

#label set
csi_list1 = ['csi'+str(selected_label.iloc[i].year)+str(selected_label.iloc[i].month).zfill(2)+str(selected_label.iloc[i].day)+str(selected_label.iloc[i].hour).zfill(2)+'03' for i in range(len(selected_label))]
selected_label = selected_label.set_index([csi_list1])
label = selected_label.total_people

# SVM
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

model = SVC(kernel='linear', C=1.0, random_state=0)
#model = SVC(kernel='rbf', C=1.0, random_state=0, gamma=0.10)
#model = Perceptron(tol=1e-3, random_state=0)
model_data_label = csi_summ.join(label)

model_label =model_data_label['total_people'].values.astype('int').astype('str')
model_data = model_data_label.drop('total_people',axis=1).values

# CrossVal
from sklearn.model_selection import KFold
cv = KFold(10,shuffle=True, random_state=12)
scores = np.zeros(10)
for i, (train_index, test_index) in enumerate(cv.split(model_data)):
    X_train = model_data[train_index]
    y_train = model_label[train_index]
    X_test = model_data[test_index]
    y_test = model_label[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores[i]= np.round(sum(y_pred==y_test)/len(y_test),decimals=2)
np.mean(scores)

##
a=np.squeeze(arr_abs2)
plt.plot(a)
plt.show()
# plot1 : abs

print("CSI:to array")
list_abs = []
for i in range(len(dict_csi)):
    abs_val = np.mean(np.abs(dict_csi[i+1][1]),axis=2)
    #abs_val = np.mean(np.abs(np.angle(dict_csi[i+1][1])),axis=2)
    if abs_val.shape==(ntx,nrx):
        list_abs.append(abs_val)
arr_abs = np.stack(list_abs, axis=-1)

print("CSI: plot")
for i in range(ntx):
    for j in range(nrx):
        plt.plot(df_sc.index,arr_abs[i,j,:],label=str((i,j)))
        plt.legend(bbox_to_anchor=(1.01, 1.01))



# plot3 : subcarrier plot

print("CSI:subcarrier") 

max_abs = 30

abs_list = []
ph_list = []
for i in range(len(data_csi)):
    abs_list.append(np.abs(data_csi.iloc[i].csi_scaled))
    ph_list.append(np.angle(data_csi.iloc[i].csi_scaled))
abs_arr = np.array(abs_list)
ph_arr = np.array(ph_list)

abs_int = (abs_arr / max_abs * 255).astype('uint8')
abs_img = np.zeros((abs_int.shape[0],abs_int.shape[3],3),'uint8')
for i in range(abs_int.shape[0]):
    for j in range(abs_int.shape[3]):
        abs_img[i][j] = abs_int[i,0,:,j]

im = Image.fromarray(abs_img)
im.save('im.jpeg')