function gb = origin_gabor(X, Y, sigma, theta, omega)
		%sigma = scale (in paper root45)
		%theta = orientation (in paper 6)
		%omega = frequency (in paper 1/12)
        %x,y = window size(in paper 33,33)
   
       filter_radius = floor(X/2);
       [x, y] = meshgrid(-filter_radius:filter_radius, -filter_radius:filter_radius);
        
        
        
		
        e = exp(1); 
            
        r = 1; %constant (values ~=1 -> eliptical filter)

        R1 = x.*cos(theta) + y.*sin(theta);
        R2 =-x.*sin(theta) + y.*cos(theta);

        expFactor = -1/2 * ( (R1/sigma).^2 + (R2/(r*sigma)).^2  );

        gauss = 1 / ( sqrt(r*pi)*sigma) ;
        gauss =  gauss .* e.^expFactor;

        gaborReal = gauss .* cos(omega*R1);
        gaborImag = gauss .* sin(omega*R1);

        kernel = gaborReal + gaborImag*1i;
        gb = kernel'; 
end