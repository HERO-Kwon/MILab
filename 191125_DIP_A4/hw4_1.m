% image read
im = im2double(imread('pepper.png'));

% 1. RGB to HSI conversion
im_hsi = toHSI(im);
% plot
figure(1);
subplot(1,3,1);
imshow(im_hsi(:,:,1)./360);
title('Hue');
subplot(1,3,2);
imshow(im_hsi(:,:,2));
title('Saturation');
subplot(1,3,3);
imshow(im_hsi(:,:,3));
title('Intensity');

% 2. HSI to RGB conversion
im_rgb2 = toRGB(im_hsi);
figure(2);
subplot(1,2,1);
imshow(im);
title('Original Image');
subplot(1,2,2);
imshow(im_rgb2);
title('Converted from HSI image');
