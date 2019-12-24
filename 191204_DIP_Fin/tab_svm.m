clear all
clc

% read data from csv file
data_tbl = readtable('C:\Data\PalmVein\PolyU-M\Multi Spectral\diff_OBNLMcanny1re_image200.csv');
disp('Read Done');

%target = 'Red';
target_id = 200;

%target_tbl = data_tbl(strcmp(data_tbl.Var1,target),:);
target_tbl = data_tbl(data_tbl.Var1 <= target_id,:);

imgs = target_tbl{:,3:end};
lab1 = target_tbl{:,1};
lab2 = cellfun(@(x) x(:,1), target_tbl{:,2});
labels = strcat(string(lab1),'_',lab2);

cols_pca = 20;

indices = crossvalind('Kfold',labels,2);
cp = classperf(labels);

% show sample image
im_s = target_tbl{1,3:end};
im_sre = reshape(im_s,[128,128]);
imshow(im_sre)

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



