clear;

img1_red = imread('tr1.jpg');
img2_blue = imread('tb1.jpg');

diffimage = diffimage(img1_red,img2_blue,0.5);
%classifier
gaborcanny = simplegabor_and_canny(diffimage);

figure(1);
subplot(1,6,1);
imshow(gaborcanny(:,:,1));
subplot(1,6,2);
imshow(gaborcanny(:,:,2));
subplot(1,6,3);
imshow(gaborcanny(:,:,3));
subplot(1,6,4);
imshow(gaborcanny(:,:,4));
subplot(1,6,5);
imshow(gaborcanny(:,:,5));
subplot(1,6,6);
imshow(gaborcanny(:,:,6));

figure(3);
Ma = max(gaborcanny,[],3);
imshow(Ma);
se1 = strel('line',3,0); %change(canny parameter)
T = edge(real(Ma),'Canny',0.5);
BW2 = imdilate(T,se1);
figure(4);
imshow(BW2);