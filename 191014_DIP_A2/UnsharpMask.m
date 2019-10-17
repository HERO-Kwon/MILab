function output = UnsharpMask (I, I_blur, k) %
% where I is and input image, and I_blur is blurred image and k is boosting factor

% apply unsharp masking filter
output = I + k .* (I - I_blur);
    
end 
% Complete the remaining part