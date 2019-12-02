function output = toHSI(im_rgb)

im_hsi = zeros(size(im_rgb));
[size_w,size_h] = size(im_rgb(:,:,3));
sum_rgb = sum(im_rgb,3);

%I
im_hsi(:,:,3) = (1/3) .* sum_rgb;
%S
im_hsi(:,:,2) = ones(size(im_rgb(:,:,2))) - 3 ./ (sum_rgb+0.000001) .* min(im_rgb,[],3);
%H
for i = 1:size_w
    for j = 1:size_h
        p_r = im_rgb(i,j,1);
        p_g = im_rgb(i,j,2);
        p_b = im_rgb(i,j,3);
        nume = 1/2*((p_r-p_g) + (p_r-p_b));
        denom = ((p_r-p_g)^2 + ((p_r-p_b)*(p_g-p_b)))^(1/2);
        h = acosd( nume / (denom+0.000001) );
        if p_g >= p_b
            im_hsi(i,j,1) = h;
        else
            im_hsi(i,j,1) = 360 - h;
        end
    end
end

output = im_hsi;
end

