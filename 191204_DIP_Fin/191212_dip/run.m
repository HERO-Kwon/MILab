clear;

img1_red = imread('train98_1_red.jpg');
img2_blue = imread('train98_1_blue.jpg');

diffimage = diffimage(img1_red,img2_blue,0.5);
gaborcanny = gaborcanny(diffimage);

figure(1);
imshow(gaborcanny);