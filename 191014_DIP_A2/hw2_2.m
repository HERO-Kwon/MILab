% image read
im = im2double(imread('cat_crop.png'));
% adjust intensity to max 255
im = im(:,:,1) .* 255;

% parameters
std_list=[7, 12, 17];
filter_list = [20, 1, 2, 10];

% 1. add Gaussian noise
figure(1);
subplot(2,2,1);
imshow(im ./ 255);
title('input image');
im_Trs = cell(3,1);
% calculate noise
for i = 1:length(std_list)
    im_Tr = GaussianNoise(im, std_list(i));
    % store image matrix
    im_Trs{i} = im_Tr;
    subplot(2,2,i+1);
    % show image correspoiding to std value
    imshow(im_Tr ./ 255);
    title(['Noise: std = ',num2str(std_list(i))]);
end

% 2. filter noisy images
for i = 1:length(std_list)
    im_Tr = im_Trs{i};
    % plot noisy image
    figure(1+i);
    for j = 1:length(filter_list)
        % apply filter
        im_b = BilateralFilter(im,filter_list(j),filter_list(j),5);

        % plot filtered image
        subplot(2,2,j);
        imshow(im_b ./ 255);
        title(['Noise: std = ',num2str(std_list(i)),' / Filter: std = ',num2str(filter_list(j))]);
    end
end