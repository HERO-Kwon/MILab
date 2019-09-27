
%2-1 Load image and Display histogram
% load image
im = im2double(imread('input3.png'));
% fig 1: histogram of the input image
figure(1);
imhist(im);
title(['2-1. Histogram of Input Image']);

%2-2 implement histogram equalization function;

% image size
size_im = size(im);
MN = size_im(1) * size_im(2);
% make 256bins of histogram
hist = imhist(im,256);
L = length(hist);
% calculate pdf,cdf
pdf = hist ./ MN;
cdf = cumsum(pdf);
% histogram equalized array
s_arr = round((L-1).*cdf);
% map input image to histogram-equalized image
output = zeros(size(im));
for i = 1:L
    mask = (im >= (i-1)/L) & (im <= i/L);
    output = output + mask .* (s_arr(i)/L);
end

% fig 2: histogram of the output image
figure(2);
imhist(output);
title(['2-2. Histogram of Output Image']);
% fig 3: Output image
figure(3);
imshow(output);
title(['2-3. Output Image']);

