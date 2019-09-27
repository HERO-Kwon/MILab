im = im2double(imread('input2.png'));

gamma = [0.2, 0.4, 2.5, 5];

figure(1);
for i = 1:length(gamma)
    hold on;
    x_axis = [0:1/255:1];
    f_Tr =  PowerLawTr(x_axis, gamma(i));
    plot(x_axis, f_Tr);
    legend({'\gamma = 0.2', '\gamma = 0.4', '\gamma = 2.5', '\gamma = 5'}, 'Location','northwest');
end

figure(2);
for i = 1:length(gamma)
    im_Tr = PowerLawTr(im, gamma(i));
    subplot(2,2,i);
    imshow(im_Tr);
    title(['\gamma = ',num2str(gamma(i))]);
end

figure(3);
for i = 1:length(gamma)
    im_Tr = PowerLawTr(im, gamma(i));
    subplot(2,2,i);
    imhist(im_Tr);
    title(['\gamma = ',num2str(gamma(i))]);
end