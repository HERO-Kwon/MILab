
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Made by HERO Kwon                     %
% 2018-02-25                            %
% Eigenfaces algorithms using ORLDB     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clf('reset')

% known variables

height = 56;
width = 46;

n_eigface = 100;

% Read ORLDB data 

%face_data = dir('D:\Data\ORLDB\orl*.bmp');
face_data = dir('/MATLAB Drive/ORLDB/orl*.bmp');
%face_directory = ('D:\Data\ORLDB\');
face_directory = ('/MATLAB Drive/ORLDB/');

expression = '.*orl(?<person>\d\d)(?<number>\d\d).*';

for i = 1:length(face_data)
    name_info = regexp(face_data(i).name,expression,'names');
    img = imread(strcat(face_directory,face_data(i).name));
 
    face_data(i).data = img;
    face_data(i).person = name_info.person;
    face_data(i).number = str2num(name_info.number);
end

% Separate train and test data

train_idx = logical([]);
test_idx = logical([]);

for i = 1:ceil(length(face_data)/10)
    face_person = face_data(10*i-9:10*i);
    [train_idx_p,test_idx_p] = crossvalind('HoldOut',length(face_person),0.5);

    train_idx = cat(1,train_idx,train_idx_p);
    test_idx = cat(1,test_idx,test_idx_p);
end

train_data = face_data(train_idx);
test_data = face_data(test_idx);
 
% Make Eigenfaces

sum_imgs = zeros(height,width,'double');

for i = 1:length(train_data)
    sum_imgs = sum_imgs + double(face_data(i).data);
end
mean_imgs = double(sum_imgs) ./ length(face_data);

for i = 1:length(train_data)
    sigma_mat = double(train_data(i).data) - mean_imgs;
    sigma{i} = sigma_mat(:);
    mat_A(:,i) = sigma{i};
end

mat_L = transpose(mat_A) * mat_A;
[eig_vec,eig_lam] = eigs(mat_L,n_eigface);

for i = 1:n_eigface
    eig_face{i} = zeros(height*width,1);
    for k = 1:length(train_data)
        eig_face{i} = eig_face{i} + ( eig_vec(k,i) .* sigma{k} );
    end
end

% Classify Test Data

person_group = findgroups(cellstr(char(test_data.person)));

for i = 1:length(test_data)
    mean_newface = (double(test_data(i).data) - mean_imgs);
    sigma_new{i} = mean_newface(:);
end

% Eigenface decomposion of test data
for i = 1:length(test_data)
    sigma_f{i} = zeros(height*width,1);
    for k = 1:n_eigface
        weight_test(k) = transpose(eig_face{k}) * sigma_new{i};
        sigma_f{i} = sigma_f{i} +  weight_test(k) * eig_face{k};
    end
    weights_test{i} = weight_test;
end

for i = 1:length(test_data)
    err_class = [];
    % Search Closest face
    for k = 1:length(test_data) %length(test_data)
        err_class(k) = norm(weights_test{i} - weights_test{k});
    end

    test_data(i).err_class = err_class;
    test_data(i).err_class_not0 = err_class(err_class ~= 0);
end

% EER

err_allval = [test_data.err_class_not0];
err_sampled = datasample(err_allval,200);
fnfp = [];

for i = 1 : length(err_sampled)
    thres_person = err_sampled(i);
    
    % make confusion matrix
    conf_mat = [];
    for k = 1:length(test_data)
        err_underthres = (test_data(k).err_class_not0 < thres_person );
        person_group_k = cat(1,person_group(1:k-1),person_group(k+1:length(person_group)));
        test_data(k).predicted_person = splitapply(@max,err_underthres',person_group_k);

        person_array = zeros(40,1);
        person_number = str2num(test_data(k).person);
        person_array(person_number) = 1;

        g1 = test_data(k).predicted_person;
        g2 = logical(person_array);

        conf_mat = cat(3,conf_mat,flipud(fliplr(confusionmat(g1,g2))));
    end

    sum_conf = sum(conf_mat,3);
    sum_n = sum(sum_conf,1);

    fn_rate = sum_conf(2,1) / sum_n(1) * 100;
    fp_rate = sum_conf(1,2) / sum_n(2) * 100;
    acc = (sum_conf(1,1) + sum_conf(2,2)) / sum(sum_n);

    fnfp = cat(1,fnfp,[thres_person, fn_rate, fp_rate, fn_rate+fp_rate, acc]);
end

fnfp(:,1) = fnfp(:,1) / max(fnfp(:,1));
fnfp = sortrows(fnfp,1);
distance_fnfp = abs(fnfp(:,2)-fnfp(:,3));

eer = fnfp(distance_fnfp==min(distance_fnfp),2)
acc_at_eer = fnfp(fnfp(:,4)==min(fnfp(:,4)),5)

figure(1)
hold on
line(fnfp(:,1) / max(fnfp(:,1)),fnfp(:,2),'Color','red')
line(fnfp(:,1) / max(fnfp(:,1)),fnfp(:,3),'Color','blue')
line(fnfp(:,1) / max(fnfp(:,1)),fnfp(:,4),'Color','green')
hold off


% Distribution curve


function dist_tf = dist_curve(data_str)

% dist_true : distibution of genuine data
% dist_false : distribution of imposter data
% data_group : 

dist_true = [];
dist_false = [];

for i = 1:length(test_data)
    person_number = str2num(test_data(i).person);
    dist_true = cat(1,dist_true,test_data(i).err_class(person_group==person_number)');
    dist_false = cat(1,dist_false,test_data(i).err_class(person_group~=person_number)');
end

max_err = max([test_data.err_class]);

dist_true_norm = dist_true(dist_true ~= 0) ./ max_err;
dist_false_norm = dist_false(dist_false ~= 0) ./ max_err;

end

figure(2)
hold on

hist_true = histogram(dist_true_norm);
hist_true.Normalization = 'probability';
hist_true.DisplayStyle = 'stairs';
hist_true.NumBins = 100;

hist_false = histogram(dist_false_norm);
hist_false.Normalization = 'probability';
hist_false.DisplayStyle = 'stairs';
hist_false.NumBins = 100;

hold off
