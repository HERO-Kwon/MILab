
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Made by HERO Kwon                     %
% 2018-02-26                            %
% Resizes and Modifies scale as ORLDB   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% small images

simg_data = dir('D:\Data\gsearch_small_image\*.jpg');
simg_directory = ('D:\Data\gsearch_small_image\');

for i = 1:length(simg_data)
    simg = imread(strcat(simg_directory,simg_data(i).name));
    simg_data(i).data = simg;

    simg_resize = imresize(simg,[56,46]);
    simg_gray = rgb2gray(simg_resize);

    simg_name = strcat('simg_resized_',string(i),'.bmp');

    imwrite(simg_gray,char(simg_name),'bmp');
end