% Lefteye_1mNIR
% Righteye_1mNIR
% 
% Lefteye_1mVIS
% Righteye_1mVIS
% 
% Lefteye_60mNIR
% Righteye_60mNIR
% 
% Lefteye_60mVIS
% Righteye_60mVIS
% 
% Lefteye_100mNIR
% Righteye_100mNIR
% 
% Lefteye_100mVIS
% Righteye_100mVIS
% 
% Lefteye_150mNIR
% Righteye_150mNIR
% 
% Lefteye_150mVIS
% Righteye_150mVIS
%----------------------------------------------



% Lefteye_150mVIS=zeros(2,100);
% Righteye_150mVIS=zeros(2,100);


for i = 100
    fn=sprintf('/home/han/Desktop/LDHF-DB/LDHF-DB/150mVIS/%04d_150_d.JPG',i);
    im(:,:)=double(rgb2gray(imread(fn)));
    [Lefteye_150mVIS Righteye_150mVIS]=pclick(im,Lefteye_150mVIS,Righteye_150mVIS,i);
    clear im;
    i
end

