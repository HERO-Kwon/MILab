function output = Resize(I,SCALE)
%RESIZE Summary of this function goes here
%   Detailed explanation goes here

mat_size = size(I);
% sampling by ratio
row_vec = 1:SCALE:mat_size(1);
col_vec = 1:SCALE:mat_size(2);
sampled_im = I(row_vec,col_vec);

output = sampled_im;

end

