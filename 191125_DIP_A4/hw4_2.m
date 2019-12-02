% image read
im = im2double(imread('cameraman.png'));
% change scale to maximum 255
im = rgb2gray(im) .* 255;

% Pseudo coloring
[size_w,size_h] = size(im);
pc_im = zeros([size_w size_h 3]);
pc_rgb = [54 21 203 16;128 154 213 233;33 233 62 59];

for i = 1:size_w
    for j = 1:size_h
        % get intensity from the pixel of gray image
        intensity = fix(im(i,j)/72);
        switch intensity
            % set color according to intensity value
            case 0
                pc_im(i,j,:) = pc_rgb(:,1);
            case 1
                pc_im(i,j,:) = pc_rgb(:,2);
            case 2
                pc_im(i,j,:) = pc_rgb(:,3);
            case 3
                pc_im(i,j,:) = pc_rgb(:,4);
        end
    end
end

% Plot
figure(1);
subplot(1,2,1);
imshow(im./255);
title('Original Image');
subplot(1,2,2);
imshow(pc_im./255);
title('pseudo-colored Image');
