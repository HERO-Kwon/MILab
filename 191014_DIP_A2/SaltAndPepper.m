function output = SaltAndPepper(img, ND)
% Add salt and pepper noise to image of certain density
% ND = noise density, if ND is 0.2, noisy image has 20% noise

% make noise vector
noise_size = size(img);
sp_vec = ones(noise_size(1) * noise_size(2),1);
% sample vector index according to noise points
sp_noise_idx = randsample(length(sp_vec),round(length(sp_vec)*ND));
% separate white and black noise index
sp_noise0 = randsample(sp_noise_idx,round(length(sp_noise_idx)*0.5));
sp_noise1 = setdiff(sp_noise_idx,sp_noise0);
% input noise value
sp_vec(sp_noise0) = 0;
sp_vec(sp_noise1) = 255;
% apply noise value to image
sp_noise = reshape(sp_vec,noise_size);
% return
output = img .* sp_noise;

end 
% Complete the remaining part
