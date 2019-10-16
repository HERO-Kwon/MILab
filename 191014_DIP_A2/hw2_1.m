im = im2double(imread('input1.jpg'));
im = im(:,:,1) .* 255;

std_list=[7, 12, 17];
filter_list = [1, 3, 5];

figure(1);
subplot(2,2,1);
imshow(im ./ 255);
title('input image');
im_Trs = cell(3,1);
for i = 1:length(std_list)
    im_Tr = GaussianNoise(im, std_list(i));
    im_Trs{i} = im_Tr;
    subplot(2,2,i+1);
    imshow(im_Tr ./ 255);
    title(['Noise: std = ',num2str(std_list(i))]);
end

figure(2);
g_filters = cell(3,1);
for i = 1:length(filter_list)
    g_filter = GaussianFilter(im,filter_list(i));
    g_filters{i} = g_filter;
    subplot(2,2,i);
    imshow(g_filter,[min(min(g_filter)),max(max(g_filter))]);
    title(['Filter: std = ',num2str(filter_list(i))]);
end



for i = 1:length(std_list)
    im_Tr = im_Trs{i};
    
    figure(2+i);
    subplot(2,2,1);
    imshow(im_Tr ./ 255);
    title(['Noise: std = ',num2str(std_list(i))]);
    for j = 1:length(filter_list)
        g_filter = g_filters{j};
        w = conv2(im_Tr,g_filter,'same');
        
        subplot(2,2,j+1);
        imshow(w ./ 255);
        title(['Filter: std = ',num2str(filter_list(j))]);
    end
end