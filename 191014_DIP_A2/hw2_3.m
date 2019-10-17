% image read
im = im2double(imread('input1.jpg'));
% adjust intensity to max 255
im = im(:,:,1) .* 255;

% parameters
noise_list=[0.1, 0.2, 0.3, 0.4, 0.5];
filter_list = [3, 5, 7];

% 1. add Salt&Pepper noise
figure(1);
subplot(2,3,1);
imshow(im ./ 255);
title('input image');
im_Trs = cell(length(noise_list),1);
% calculate noise
for i = 1:length(noise_list)
    im_Tr = SaltAndPepper(im, noise_list(i));
    % store image matrix
    im_Trs{i} = im_Tr;
    subplot(2,3,i+1);
    % show image correspoiding to noise value
    imshow(im_Tr ./ 255);
    title(['ND : ',num2str(noise_list(i))]);
end

% 2. filter noisy images
for i = 1:length(noise_list)
    im_Tr = im_Trs{i};
    % plot noisy image
    figure(1+i);
    subplot(2,2,1);
    imshow(im_Tr ./ 255);
    title(['ND : ',num2str(noise_list(i))]);
    
    for j = 1:length(filter_list)
        % apply filter
        im_m = MedianFilter(im,filter_list(j));
        % plot filtered image
        subplot(2,2,j+1);
        imshow(im_m ./ 255);
        title(['ND : ',num2str(noise_list(i)),' / Filter : [',num2str(filter_list(j)),',',num2str(filter_list(j)),']']);
    end
end