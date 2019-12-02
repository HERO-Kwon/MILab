% image read
im = im2double(imread('image1.png'));
% adjust intensity to max 255
im = im(:,:,1) .* 255;

% parameters
noise_list=[0.1, 0.2, 0.3];
filter_list = [3, 5];

% plot original image
figure(1);
subplot(1,4,1);
imshow(im ./ 255);
title('input image');
im_Trs = cell(length(noise_list),1);
% 1. add Salt&Pepper noise
for i = 1:length(noise_list)
    im_Tr = SaltAndPepper(im, noise_list(i));
    % store image matrix
    im_Trs{i} = im_Tr;
    subplot(1,4,i+1);
    % show image correspoiding to noise value
    imshow(im_Tr ./ 255);
    title(['ND : ',num2str(noise_list(i))]);
end

% 2. filter noisy images
for i = 1:length(noise_list)
    im_Tr = im_Trs{i};
    % plot noisy image
    figure(2);
    subplot(3,3,3*(i-1)+1);
    imshow(im_Tr ./ 255);
    title(['ND : ',num2str(noise_list(i))]);
    
    for j = 1:length(filter_list)
        % apply filter
        % 3. improved median filter
        im_m = ImprovedMedianFilter(im,filter_list(j));
        %im_m = MedianFilter(im,filter_list(j));
        % plot filtered image
        subplot(3,3,3*(i-1)+(j+1));
        imshow(im_m ./ 255);
        title(['ND : ',num2str(noise_list(i)),' / Filter : [',num2str(filter_list(j)),',',num2str(filter_list(j)),']']);
    end
end


