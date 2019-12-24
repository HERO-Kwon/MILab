clear all
clc

% read data from csv file
data_tbl = readtable('C:\Data\PalmVein\PolyU-M\Multi Spectral\imgs.csv');

target = 'Red';
target_id = 100;

target_tbl = data_tbl(strcmp(data_tbl.Var1,target),:);
target_tbl = target_tbl(target_tbl.Var2 <= target_id,:);

imgs = target_tbl{:,4:end};
labels = target_tbl{:,2};

cols_pca = 2;

indices = crossvalind('Kfold',labels,2);
cp = classperf(labels);

for i = 1:2
    test = (indices == i); 
    train = ~test;
    train_pca = pca(imgs(train,:)','NumComponents',cols_pca);
    train_lab = labels(train,:);
    test_pca = pca(imgs(test,:)','NumComponents',cols_pca);
    test_lab = labels(test,:);
    
    SVMModel = fitcecoc(train_pca,train_lab);
    [pred_lab,score] = predict(SVMModel,test_pca);
    acc = sum(test_lab==pred_lab) / length(test_lab);
    
    disp([i,acc])
end