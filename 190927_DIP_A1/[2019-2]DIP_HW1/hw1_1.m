c = im2double(imread('input1.png'));

cTr = PiecewiseLinearTr(c, [0,1], [1,0]);
cTr2 = PiecewiseLinearTr(c, [0 .25 .5 .75 1],[0 .75 .25 .5 1]);

figure(1);
x_axis = [0:1/255:1];
subplot(1,2,1);
plot(x_axis, PiecewiseLinearTr(x_axis, [0,1], [1,0]));
subplot(1,2,2);
plot(x_axis, PiecewiseLinearTr(x_axis, [0 .25 .5 .75 1],[0 .75 .25 .5 1]));

figure(2);
subplot(1,2,1);
imshow(cTr);
subplot(1,2,2);
imshow(cTr2);

figure(3);
subplot(1,2,1);
imhist(cTr);
subplot(1,2,2);
imhist(cTr2);