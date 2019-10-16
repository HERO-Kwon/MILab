function output = GaussianFilter (I,SIGMA) %
% where I is and Gaussian noise images(std : 7,12,17),, SIGMA is standard deviation of Gaussian function 
% Each dimension of filter is determined automatically by 2*ceil(2*sigma)+1

% filter dimension
G_len = 2*ceil(2*SIGMA)+1;
% make zero matrix corresponding to filter size
G_mat = zeros(G_len,G_len);
% x,y range
range = linspace(-G_len/2,G_len/2,G_len);
% calculate filter matrix values
for x = 1:G_len
    for y = 1:G_len
        % Gaussian distribution
        G_sigma =   1./(2*pi*SIGMA*SIGMA).*exp(-1 .* (range(x)*range(x) + range(y)*range(y))./(2*SIGMA*SIGMA));
        G_mat(x,y) = G_sigma;
    end
end
% return filter matrix
output = G_mat;

% Complete the remaining part