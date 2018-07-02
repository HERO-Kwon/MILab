import os
import pandas as pd
import shutil
import re

# data path
data_path = 'D:\\Matlab_Drive\\Data\\Severance\\H4010'
label_path = 'D:\\Matlab_Drive\\Data\\Severance\\label'

label = pd.read_csv(label_path + '\\H4010.csv',dtype=object)

# data save path
nolabel_path = 'D:\\Matlab_Drive\\Data\\Severance\\data_nolabel'
withlabel_path = 'D:\\Matlab_Drive\\Data\\Severance\\data_withlabel'

# file seacher
res_df = pd.DataFrame(columns=['Num','Date','vod','vos','tn','Files'])
for root, dirs, files in os.walk(data_path):
    #filter folder contains jpg file
    jpg_files = [s for s in files if ".jpg" in s]

    if(jpg_files):  
        # vigit date
        date = os.path.basename(root)
        # patient number
        num = os.path.basename(root[:-7])

        # filter lael
        values = label[(label['진료일']==date) & (label['등록번호']==num)]
        
        if(values.empty):
            # no label data
            for fname in files:
                full_fname = os.path.join(root, fname)
                shutil.copy2(full_fname,nolabel_path)
        else:
            if (len(files)==6):
                files.sort()
                fnames = [files[0],files[3]]
                for fname in fnames:
                    full_fname = os.path.join(root, fname)
                    shutil.copy2(full_fname,withlabel_path)
            else:
                fnames = files
                for fname in fnames:
                    full_fname = os.path.join(root, fname)
                    shutil.copy2(full_fname,withlabel_path)
            so = values['S&O'].values
            
            # extract vod,vos,tn info
            vod = re.search('[vV][oO][dD]\D*(\d+\.*\d+)\D*',so[0])
            vos = re.search('[vV][oO][sS]\D*(\d+\.*\d+)\D*',so[0])
            tn = re.search('[tT][nN]\D*(\d*.*/.*\d*)\D*',so[0])

            # save result
            res_val = pd.Series()

            res_val['Num']=num
            res_val['Date']=date
            res_val['Files']=fnames
            try:
                res_val['vod']=vod.group(1)
            except:
                res_val['vod']=''
            try:
                res_val['vos']=vos.group(1)
            except:
                res_val['vos']=''
            try:
                res_val['tn']=re.sub(' ','',tn.group(1))
            except:
                res_val['tn']=''
            # merge result
            res_df = res_df.append(res_val,ignore_index=True)

res_df.to_csv('res_df.csv')