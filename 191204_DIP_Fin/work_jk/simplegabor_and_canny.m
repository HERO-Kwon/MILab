function simplegabor_and_canny = simplegabor_and_canny(image)

x = 31; %change
y = 31; %change
sigma = 2; %change
omega = x/(128/31); %change

theta   = 0; %fix
N = 6; %fix


se1 = strel('line',3,0); %change(canny parameter)

img_out = zeros(size(image,1),size(image,2), N);
BW2 = zeros(size(image,1),size(image,2), N);
GB = zeros(31,31, N);
for n=1:N
    %gabor filter
    gb = origin_gabor(x, y, sigma, theta, omega);
    GB(:,:,n) = gb;
    QCF = f_Quanta_Gabor(gb,1,2); 
    
    
    img_out(:,:,n) = imfilter(image, QCF, 'symmetric');
     
    %canny edge detection 
    T = edge(real(img_out(:,:,n)),'Canny',0.5);
    BW2(:,:,n) = imdilate(T,se1);
   
    
    theta = theta + 30;
end

%simplegabor_and_canny = BW2;
%simplegabor_and_canny = GB;
simplegabor_and_canny = img_out;


