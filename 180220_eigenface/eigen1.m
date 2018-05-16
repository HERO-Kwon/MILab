
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Made by HERO Kwon                     %
% 2018-02-25                            %
% Eigenfaces algorithms using ORLDB     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% known variables

height = 56;
width = 46;

thres_face = 5 * 10^12;

% Read ORLDB data 

face_data = dir('D:\Data\ORLDB\orl*.bmp');
face_directory = ('D:\Data\ORLDB\');

expression = '.*orl(?<person>\d\d)(?<number>\d\d).*';

for i = 1:length(face_data)
    name_info = regexp(face_data(i).name,expression,'names');
    img = imread(strcat(face_directory,face_data(i).name));
 
    face_data(i).data = img;
    face_data(i).person = str2num(name_info.person);
    face_data(i).number = str2num(name_info.number);
end

simg_data = dir('D:\Data\small_image_resized\simg*.bmp');
simg_directory = ('D:\Data\small_image_resized\');

simg_expression = '.*simg_resized_(?<number>\d+).*';
% simg_resized_1.bmp
for i = 1:length(simg_data)
    simg = imread(strcat(simg_directory,simg_data(i).name));
    simg_name_info = regexp(simg_data(i).name,simg_expression,'names');

    simg_data(i).data = simg;
    simg_data(i).person = 0;
    simg_data(i).number = str2num(simg_name_info.number);
end


% Separate train and test data

[train_idx,test_idx] = crossvalind('HoldOut',length(face_data),0.1);

train_data = face_data(train_idx);
test_data = [face_data(test_idx)]; %;simg_data];

% Make Eigenfaces

sum_imgs = zeros(height,width,'int16');

for i = 1:length(train_data)
    sum_imgs = sum_imgs + int16(face_data(i).data);
end
mean_imgs = double(sum_imgs) ./ length(face_data);

for i = 1:length(train_data)
    sigma_mat = double(train_data(i).data) - mean_imgs;
    sigma{i} = sigma_mat(:);
    mat_A(:,i) = sigma{i};
end

mat_L = transpose(mat_A) * mat_A;
[eig_vec,eig_lam] = eig(mat_L);

for i = 1:length(train_data)
    eig_face{i} = zeros(height*width,1);
    for k = 1:length(train_data)
        eig_face{i} = eig_face{i} + ( eig_vec(k,i) .* sigma{k} );
    end
end

% Eigenface decomposition of training data 

for i = 1:length(train_data)
    for k = 1:length(train_data)
        weight_train(k) = transpose(eig_face{k}) * sigma{i};
    end
    weights_train{i} = weight_train;
end

% Classify Test Data

for i = 1:length(test_data)
    mean_newface = (double(test_data(i).data) - mean_imgs);
    sigma_new{i} = mean_newface(:);
end

% Eigenface decomposion of test data
for i = 1:length(test_data)
    sigma_f{i} = zeros(height*width,1);
    for k = 1:length(train_data)
        weight_test(k) = transpose(eig_face{k}) * sigma_new{i};
        sigma_f{i} = sigma_f{i} +  weight_test(k) * eig_face{k};
    end

    % Search Closest face

    for k = 1:length(train_data)
        err_class(k) = norm(weights_train{k} - weight_test);
    end

    [val,ind] = min(err_class);
    test_data(i).predicted_person = train_data(ind).person;
    
end

% Projection into face space
for i = 1:length(test_data)
    err_face(i) = norm(sigma_new{i} - sigma_f{i});
    test_data(i).is_person = ( err_face(i) < thres_face );
end

% Calculate Accuracy

acc = sum( [test_data.person] == [test_data.predicted_person] ) / length(test_data);
