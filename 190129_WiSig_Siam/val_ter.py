def EER_Curve(tf_list,err_values,arr_thres):

    print("EER Curve")

    eer_df = pd.DataFrame(columns = ['thres','fn','fp','tn','tp'])
    

    for i, thres in enumerate(set(arr_thres)):
        predicted_tf = [e <= thres for e in err_values]
        
        tn, fp, fn, tp = confusion_matrix(tf_list,predicted_tf).ravel()

        eer_ser = {'thres':thres,'tn':tn,'fp':fp,'fn':fn,'tp':tp}
        eer_df = eer_df.append(eer_ser,ignore_index=True)
        
        curr_percent = 100 * (i+1) / len(arr_thres)
        if (curr_percent % 10)==0 : print(int(curr_percent),end="|")

    eer_df_graph = eer_df.sort_values(['thres'])
    eer_df_graph.fn = eer_df_graph.fn / max(eer_df_graph.fn) * 100
    eer_df_graph.fp = eer_df_graph.fp / max(eer_df_graph.fp) * 100
    eer_df_graph.te = eer_df_graph.fn + eer_df_graph.fp

    min_te_pnt = eer_df_graph[eer_df_graph.te == min(eer_df_graph.te)]
    min_te_val = float((min_te_pnt['fn'].values[0] + min_te_pnt['fp'].values)[0] / 2)

    plt.plot(eer_df_graph.thres,eer_df_graph.fn,color='red',label='FNR')
    plt.plot(eer_df_graph.thres,eer_df_graph.fp,color='blue',label='FPR')
    plt.plot(eer_df_graph.thres,eer_df_graph.te,color='green',label='TER')
    plt.axhline(min_te_val,color='black')
    plt.text(max(eer_df_graph.thres)*0.9,min_te_val-10,'EER : ' + str(round(min_te_val,2)))
    plt.legend()
    plt.title("EER Curve")

    plt.show()

    return(min_te_val,eer_df)

def Fe_HC(target_avg,c,s,num_eig):
    X = target_avg.T
    ones = np.ones(X.shape[1]).reshape([-1,1])
    u = np.dot(1/(c*s) * X,ones)
    A = X - np.dot(u,ones.T)
    Z = np.dot(A,A.T)
    V,W = np.linalg.eig(Z)
    return(W[:num_eig,:],u)

## Data
PATH = "/home/herokwon/mount_data/Data/Wi-Fi_HC/180_100/"

#Read Data
n_splits = 2
data_skf = ReadData(PATH,n_splits)

for splits in range(len(data_skf)):
    X,c,Xval,cval,Y,Yval = data_skf[splits]

    target_xs = np.squeeze(X)
    data_xs = np.mean(target_xs,axis=2).reshape([-1,500*6])
    target_xsv = np.squeeze(Xval)
    data_xsv = np.mean(target_xsv,axis=2).reshape([-1,500*6])
    data_y = np.squeeze(Y)#.reshape([-1,1])
    data_yv = np.squeeze(Yval)#.reshape([-1,1])

    for ii in range(10):
        alpha = TERmodel_new(0.5,0.5,data_xs,data_y,M_dict[str(ii)])
        data_test = data_xsv.dot(alpha)

        n_classes = data_test.shape[0]
        res_list = []
        t1 =time.time()
        for i in range(n_classes):
            for ti in range(n_classes):
                eer_dist = data_test[i,:] - data_test[ti,:]
                score = np.sqrt(np.sum(np.multiply(eer_dist,eer_dist)))

                score_list = [ii,splits,i,ti,int(data_yv[i][0] == data_yv[ti][0]),score]
                res_list.append(score_list)

        res_df = pd.DataFrame(res_list,columns=['ter_num','splits','img1','img2','TF','score'])
        res_df1 = res_df[res_df.img1 != res_df.img2]

        res_eer = pd.concat([res_eer,res_df1])

        print(str(ii) + '_EER Time:' + str(time.time() - t1))

def TERmodel_new(r,n,X,Y,mod_w):
    #simplified version of TER
    alpha = []
    for k in list(set(Y)):
        P = X

        mk_n = X[Y!=k].shape[0]
        mk_p = X[Y==k].shape[0]
        w_n = 1/mk_n
        w_p = 1/mk_p

        w_n = mod_w[0][0] * w_n + mod_w[0][1]
        w_p = mod_w[1][0] * w_n + mod_w[1][1]

        ones_mkn = 1*(Y!=k) *w_n
        ones_mkp = 1*(Y==k) *w_p

        W = np.zeros((len(Y), len(Y)), float)
        np.fill_diagonal(W,ones_mkn+ones_mkp)
        yk = ((r-n)*ones_mkn+(r+n)*ones_mkp).T
        ak = np.linalg.pinv((P.T).dot(W).dot(P)).dot(P.T).dot(W).dot(yk)

        alpha.append(ak)
    return(np.array(alpha).T)