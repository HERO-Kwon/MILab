% i. image read
im = im2double(imread('bird.png'));

% ii. Laplacian filtering
% make Laplacian filter mask
mask = [0 1 0;1 -4 1;0 1 0];
% apply Laplacian filter
laped_im = LaplacianFilter(im,mask);

% iii. Show Laplacian filtered image
figure(1);
subplot(1,2,1);
imshow(im);
title('Original Image');
subplot(1,2,2);
imshow(laped_im);
title('Laplacian-filtered Image');

% iii. HSI conversion
hsi_im = toHSI(im);

% iv. apply Laplacian filter to HSI intensity
laped_hsi = hsi_im;
laped_hsi(:,:,3) = LaplacianFilter(laped_hsi(:,:,3),mask);

% v. HSI to RGB
laped_rgb = toRGB(laped_hsi);
% show conversion
figure(2);
subplot(1,2,1);
imshow(im);
title('Original Image');
subplot(1,2,2);
imshow(laped_rgb);
title('Laplacian filter to HSI and converted to RGB');

% vi. show difference
% plotting
figure(3);
subplot(1,2,1);
imshow(laped_im);
title('Laplacian filter to Original Image');
subplot(1,2,2);
imshow(laped_rgb);
title('Laplacian filter to HSI and converted to RGB');
% histogram
figure(4)
d = 0:0.01:1;
subplot(1,2,1);
hist(reshape(laped_im,[],3),d);
colormap([1 0 0;0 1 0;0 0 1]);
ylim([0 200]);
subplot(1,2,2);
hist(reshape(laped_rgb,[],3),d);
colormap([1 0 0;0 1 0;0 0 1]);
ylim([0 200]);
