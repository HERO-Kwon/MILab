function output = OBNLM(input)

%params
M = 20;      % search area size (2*M + 1)^2
alpha = 3;  % patch size (2*alpha + 1)^2
h = 1;    % smoothing parameter [0-infinite].
% If you can see structures in the "Residual image" decrease this parameter
% If you can see speckle in the "denoised image" increase this parameter

offset = 100; % to avoid Nan in Pearson divergence computation
% According to the gain used by your US device this offset can be adjusted.


        img = input;
        %img = imread([pathin namein]);
        %info = imfinfo([pathin namein]);
        
        [dimxy dimt] = size(size(img));
        if ( dimt > 2)
            img = rgb2gray(img);
        end
        
        % Intensity normalization
        imgd = double(img);
        mini = (min(imgd(:)));
        imgd = (imgd - mini);
        maxi = max(imgd(:));
        imgd = (imgd / maxi) * 255;
        imgd = imgd + offset; % add offset to enable the pearson divergence computation (i.e. avoid division by zero).
        s = size(imgd);
        
        % Padding
        imgd = padarray(imgd,[alpha alpha],'symmetric');
        fimgd=bnlm2D(imgd,M,alpha,h);
        fimgd = fimgd - offset;
        imgd = imgd - offset;
        imgd = imgd(alpha+1: s(1)+alpha, alpha+1: s(2)+alpha);
        fimgd = fimgd(alpha+1: s(1)+alpha, alpha+1: s(2)+alpha);
        
        % Display
        %minds = min(imgd(:));
        %maxds = max(imgd(:));
        %figure;
        %imagesc(imgd,[minds maxds]);
        %title('Original')
        %colormap(gray);
        %colorbar;
        %figure;
        %colormap(gray);
        %imagesc(fimgd,[minds maxds]);
        %title('Denoised by Bayesian NLM')
        %colorbar;
        %figure;
        %colormap(gray);
        %speckle = abs(imgd(:,:) - fimgd(:,:));
        %imagesc(speckle);
        %title('Residual image')
        %colorbar;
        
        
        %if (  strcmp(ext , '.gif') || strcmp(ext , '.bmp') || strcmp(ext , '.jpg') || strcmp(ext , '.ppm') ||  strcmp(ext , '.png') || (strcmp(ext , '.pgm') ))
        %    fimg = uint8(fimgd);
        %    img = uint8(imgd);
        %    %speckle = uint8(speckle);
        %    
        %elseif(strcmp(ext , '.tif') || strcmp(ext , '.tiff'))

        %    fimg = fimgd/ max(imgd(:));
        %    img = imgd/ max(imgd(:)); 
            %speckle = speckle/ max(imgd(:)); 
        %end
        
        %imwrite(fimg,[pathout nout]);
        %imwrite(img,[pathout nameinnorm]);
        %imwrite(speckle,[pathout namespeckle]);

        output = fimgd;


end

