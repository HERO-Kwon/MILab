% image read
im = im2double(imread('noisy.png'));
% adjust intensity to max 255
im = im(:,:,1) .* 255;

% Fourier Transform
f_im = fftshift(fft2(im));
% shifted magnitude image
abs_f_im = abs(f_im);
% plot magnitude
figure(1);
imshow(abs_f_im / max(max(abs_f_im)) * 255);

% find peaks
row_sum = sum(abs_f_im,1);
col_sum = sum(abs_f_im,2);
% plot peaks
figure(2);
subplot(2,1,1);
[pks_r,locs_r,w_r,p_r] = findpeaks(row_sum,'SortStr','descend');
plot(row_sum)
text(locs_r(1:5),pks_r(1:5)+10^6,num2str(locs_r(1:5)'))
title('Peaks:RowSum');
subplot(2,1,2);
[pks_c,locs_c,w_c,p_c] = findpeaks(col_sum,'SortStr','descend');
plot(col_sum)
text(locs_c(1:5),pks_c(1:5)+10^5,num2str(locs_c(1:5)))
title('Peaks:ColSum');

% make filter
filter_freq = ones(size(abs_f_im));
for i = 2:length(locs_r)
    filter_freq(locs_c(i),locs_r(i)) = 0;
end

%apply filter
ff_im = f_im .* filter_freq;
abs_ff_im = abs(ff_im);
% plot frequency filtered image
figure(4);
imshow(abs_ff_im / max(max(abs_ff_im)) * 255);

%inverse fourier transform
if_im = abs(ifft2(ff_im));
% plot filtered original image
figure(5);
subplot(1,2,1);
imshow(im ./ 255);
title('Original Image');
subplot(1,2,2);
imshow(if_im ./ 255);
title('Freq. Filtered Image');
