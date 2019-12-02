function output = toRGB(im_hsi)
im_rgb = zeros(size(im_hsi));
[size_w,size_h] = size(im_hsi(:,:,2));

% calc RGB value
for i = 1:size_w
    for j = 1:size_h
        p_h = im_hsi(i,j,1);
        p_s = im_hsi(i,j,2);
        p_i = im_hsi(i,j,3);
        
        if ((p_h >= 0) && (p_h < 120))
            p_r = p_i*(1+p_s*cosd(p_h)/cosd(60-p_h));
            p_b = p_i*(1-p_s);
            p_g = 3*p_i-(p_r + p_b);
        elseif ((p_h >= 120) &&  (p_h < 240))
            p_r = p_i*(1-p_s);
            p_g = p_i*(1+p_s*cosd(p_h-120)/cosd(p_h-180));
            p_b = 3*p_i-(p_r + p_g);
        elseif ((p_h >= 240) && (p_h <= 360))
            p_g = p_i*(1-p_s);
            p_b = p_i*(1 + p_s*cosd(p_h-240)/cosd(p_h-300));
            p_r = 3*p_i-(p_g + p_b);
        end
        
        im_rgb(i,j,1) = p_r;
        im_rgb(i,j,2) = p_g;
        im_rgb(i,j,3) = p_b;        
    end
end
% output
output = im_rgb;
end

