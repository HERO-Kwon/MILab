im = im2double(imread('input1.png'));

gamma = [5, 10, 20, 40];

figure(1);
 for i = 1:length(gamma)
     hold on;
     x_axis = [0:1/255:1];
     f_Tr =  SigmoidTr(x_axis, gamma(i));
     plot(x_axis, f_Tr);
     legend({'\gamma = 5', '\gamma = 10', '\gamma = 20', '\gamma = 40'}, 'Location','northwest');
 end

 figure(2);
for i = 1:length(gamma)
    im_Tr = SigmoidTr(im, gamma(i));
    subplot(2,2,i);
    imshow(im_Tr);
    title(['\gamma = ',num2str(gamma(i))]);
end

figure(3);
for i = 1:length(gamma)
    im_Tr = SigmoidTr(im, gamma(i));
    subplot(2,2,i);
    imhist(im_Tr);
    title(['\gamma = ',num2str(gamma(i))]);
end