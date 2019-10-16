function output = BilateralFilter (I,SIGMA_d, SIGMA_r, kernel_size) %
% where I is and Gaussian noise images(std : 7,12,17), SIGMA_d and SIGMA_r are standard deviations of Bilateral function. Kernel size n is nxn box.

%padding
pad_len = floor(kernel_size/2);
I_pad = padarray(I, [pad_len pad_len]);
% size of padded image
shape_pad = size(I_pad);
% calculate Gaussian bilateral filter function
I_b = zeros(size(I));
% padded image
for r_I = 1+pad_len : shape_pad(1)-pad_len
    for c_I = 1+pad_len : shape_pad(2)-pad_len
        W_p = 0;
        I_p = 0;
        % kernel
        for r_k = r_I-pad_len:r_I+pad_len
            for c_k = c_I-pad_len:c_I+pad_len
                % calculate w value
                w = exp(-1*((r_I-r_k)^2+(c_I-c_k)^2) / (2*(SIGMA_d^2)) - ((I_pad(r_I,c_I)-I_pad(r_k,c_k))^2 / (2*(SIGMA_r^2))));
                % sum w values in the kernel
                W_p = W_p + w;
                I_p = I_p + I_pad(r_k,c_k)*w;
            end
        end
        I_b(r_I-pad_len,c_I-pad_len) = I_p / W_p;
    end
end
% return
output = I_b;
end 

