function output = ImprovedMedianFilter(I, kernel_size)
% where I is and input image, and kernel_size n is nxn size of the filter

%padding
pad_len = floor(kernel_size/2);
I_pad = padarray(I, [pad_len pad_len],0,'both');
% parameters
shape_pad = size(I_pad);
mtp_param = ceil(kernel_size*kernel_size/2);
alpha = 26;
isnoise = zeros(size(I_pad));
% Calc MTP
for r_I = 1+pad_len : shape_pad(1)-pad_len
    for c_I = 1+pad_len : shape_pad(2)-pad_len
        % NLD
        k_values = I_pad(r_I-pad_len:r_I+pad_len,c_I-pad_len:c_I+pad_len);
        nld = abs(I_pad(r_I,c_I) - k_values);
        %RLD
        v_nld = reshape(nld,1,[]);
        v_nld(mtp_param) = [];
        rld = sort(v_nld);
        %MTP
        mtp = 1/mtp_param * sum(rld(1:mtp_param));
        i_th = alpha + log2(mtp);
        % check the pixel is noisy or not
        isnoise(r_I,c_I) = I_pad(r_I,c_I) > i_th;
        % if noisy : perform median filtering
        if(isnoise(r_I,c_I))
            I_pad(r_I,c_I) = median(median(k_values));
        end        
    end
end

% return
output = I_pad(1+pad_len:shape_pad(1)-pad_len,1+pad_len:shape_pad(2)-pad_len);

end

