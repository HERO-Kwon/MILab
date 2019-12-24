

%im_s = target_tbl{1,3:end};
%im_re = abs(reshape(im_s,[128,128]));

%Fpoints = detectSURFFeatures(im_re);
%[features, validPoints] = extractFeatures(im_re,Fpoints);


featureMatrix = [];
for n = 1:200
    disp(n)
    im_s = target_tbl{n,3:end};
    im_re = abs(reshape(im_s,[128,128]));

    Fpoints = detectSURFFeatures(im_re);
    [features, validPoints] = extractFeatures(im_re,Fpoints);
    
    featurematrix{n} = [featureMatrix; features];
end