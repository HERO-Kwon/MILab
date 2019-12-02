function output = MedianFilter (I, kernel_size) %
% where I is and input image, and kernel_size n is nxn size of the filter

%padding
pad_len = floor(kernel_size/2);
I_pad = padarray(I, [pad_len pad_len]);
% size of padded image
shape_pad = size(I_pad);
% make kernel function
for r_I = 1+pad_len : shape_pad(1)-pad_len
    for c_I = 1+pad_len : shape_pad(2)-pad_len
        % get values in the filter
        k_values = I_pad(r_I-pad_len:r_I+pad_len,c_I-pad_len:c_I+pad_len);
        % input median value to padded image
        I_pad(r_I,c_I) = median(median(k_values));
    end
end 
% return
output = I_pad(1+pad_len:shape_pad(1)-pad_len,1+pad_len:shape_pad(2)-pad_len);
% Complete the remaining part