function diffimage = diffimage(imagered,imageblue,alpha)
image_red_double = double(imagered);
image_blue_double= double(imageblue);
minmax_red = minmax(image_red_double);
minmax_blue = minmax(image_blue_double);

diffimage = minmax_red - alpha*minmax_blue;
