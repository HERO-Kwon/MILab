% image read
im = im2double(imread('eye.png'));
% adjust intensity to max 255
im = im(:,:,1) .* 255;

% parameters
filter_list = [3, 11];
k_list = [1, 3];
% 2. make unsharp masking filter
for i = 1:length(k_list)
    for j = 1:length(filter_list)
        % make blurry image
        % make avg filter
        avg_filter = ones([filter_list(j),filter_list(j)]);
        % apply avg filter to image
        im_blurry = conv2(im,avg_filter,'same') ./ numel(avg_filter);

        % apply unsharp maks filter
        im_unmask = UnsharpMask(im,im_blurry,k_list(i));
        
        % plotting
        figure(i);
        % plot blurry image
        subplot(2,2,2*j-1);
        imshow(im_blurry ./ 255);
        title(['blurry image : ', '/ Filter : [',num2str(filter_list(j)),',',num2str(filter_list(j)),']']);
        
        % plot sharpened image
        subplot(2,2,2*j);
        imshow(im_unmask ./ 255);
        title(['sharpened image : ', ' / k : ',num2str(k_list(i))]);
    end    
end