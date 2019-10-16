% image read
im = im2double(imread('input1.jpg'));
% adjust intensity to max 255
im = im(:,:,1) .* 255;

% parameters
std_list=[7, 12, 17];
filter_list = [1, 3, 5];

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

% 2. make Gaussian filter
figure(2);
g_filters = cell(3,1);
for i = 1:length(filter_list)
    g_filter = GaussianFilter(im,filter_list(i));
    g_filters{i} = g_filter;
    % plot filter matrix
    subplot(2,2,i);
    % normalize intensity for visualization
    imshow(g_filter,[min(min(g_filter)),max(max(g_filter))]);
    title(['Filter: std = ',num2str(filter_list(i))]);
end

% 3. filter noisy images
for i = 1:length(std_list)
    im_Tr = im_Trs{i};
    % plot noisy image
    figure(2+i);
    subplot(2,2,1);
    imshow(im_Tr ./ 255);
    title(['Noise: std = ',num2str(std_list(i))]);
    for j = 1:length(filter_list)
        g_filter = g_filters{j};
        % apply filter using convolution function
        % padding = 'same' for set output size = input size
        w = conv2(im_Tr,g_filter,'same');
        
        % plot filtered image
        subplot(2,2,j+1);
        imshow(w ./ 255);
        title(['Filter: std = ',num2str(filter_list(j))]);
    end
end