% image read
im = im2double(imread('mandrill.png'));
% adjust intensity to max 255
im = im(:,:,1) .* 255;

% sampling parameters
sampling_factors = [2 4];

% plot original
figure(1);
subplot(1,3,1);
imshow(im ./ 255);
title('input image');

% 1. down-sampling
for i = 1:length(sampling_factors)
    sampled_im = Resize(im,sampling_factors(i));
    % plot sampled image
    subplot(1,3,1+i);
    imshow(sampled_im ./ 255,'InitialMagnification', sampling_factors(i)*100);
    title({['Sampling Factors of ',num2str(sampling_factors(i))];
    ['(',num2str(sampling_factors(i)),' x zoom)']});
end

% 2. gaussian blur
figure(2);
subplot(2,2,1);
imshow(im ./ 255);
title('input image');
% calculate noise
noise_std = 17;
im_Tr = GaussianNoise(im, noise_std);
% store image matrix
im_Trs{i} = im_Tr;
% show image correspoiding to std value
subplot(2,2,2);
imshow(im_Tr ./ 255);
title(['Noise: std = ',num2str(noise_std)]);

% down-sampling
for i = 1:length(sampling_factors)
    sampled_imTr = Resize(im_Tr,sampling_factors(i));
    % plot sampled image
    subplot(2,2,2+i);
    imshow(sampled_imTr ./ 255,'InitialMagnification', sampling_factors(i)*100);
    title({['Noise: std = ',num2str(noise_std)];
    ['Sampling Factors of ',num2str(sampling_factors(i))];
    ['(',num2str(sampling_factors(i)),' x zoom)']});
end
