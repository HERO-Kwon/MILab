function gaborcanny = gaborcanny(diffimage)
I = OBNLM(diffimage);
I = I/255;
I = imgaussfilt(I,4);

gaborArray = gabor([8],[30 60 90 120 150 180]);

gaborMag = imgaborfilt(I,gaborArray);
BW2 = zeros(size(I,1),size(I,2), 6);

%figure(1);
%subplot(2,6,1);
for p = 1:6
    %subplot(2,3,p)
    %imshow(gaborMag(:,:,p),[]);
    %theta = gaborArray(p).Orientation;
    %lambda = gaborArray(p).Wavelength;
    %title(sprintf('Orientation=%d, Wavelength=%d',theta,lambda));
    BW2(:,:,p) = (gaborMag(:,:,p));
end

 Ma = min(BW2,[],3);
 b = imsharpen(Ma,'Radius',5,'Amount',4,'Threshold',1);
 T = edge(b,'Canny');
 
 gaborcanny = T;