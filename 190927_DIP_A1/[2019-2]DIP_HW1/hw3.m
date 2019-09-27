%3-1 Load images
% load image
im = im2double(imread('input4.png'));
im_match = im2double(imread('input4_match.png'));

%3-2 Display histogram
% fig 1: histogram of the input image
figure(1);
imhist(im);
title(['3-2-1. Histogram of Input Image']);
% fig 1: histogram of the input image
figure(2);
imhist(im_match);
title(['3-2-2. Histogram of the Matching Image']);

%2-2 implement histogram matching function;
% image size
size_im = size(im);
MN = size_im(1) * size_im(2);
% make 256bins of histogram
L = 256;
hist = imhist(im,L);
hist_match = imhist(im_match,L);

% calculate pdf,cdf
pdf = hist ./ MN;
pdf_match = hist_match ./ MN;

s_k = round((L-1).*cumsum(pdf));
G_zq = round((L-1).*cumsum(pdf_match));

% map input image to histogram-equalized image
output = zeros(size(im));
for i = 1:L
    mask = (im >= (i-1)/L) & (im <= i/L);
    T_rk = s_k(i);
    
    dist = abs(G_zq-T_rk);
    [val_dist,idx_dist] = min(dist);
    
    T_sk = idx_dist;
    output = output + mask .* (T_sk/L);
end

% fig 2: histogram of the output image
figure(3);
imhist(output);
title(['3-3-1. Histogram of Output Image']);
% fig 3: Output image
figure(4);
imshow(output);
title(['3-3-2. Output Image']);