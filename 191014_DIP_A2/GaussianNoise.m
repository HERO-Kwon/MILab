function output = GaussianNoise (I,SIGMA) %
% random number from SIGMA std Gaussian distrubution
output = I + SIGMA.*randn(size(I));
end 

