function output = SigmoidTr(input, gamma)
% Returns transformed image by sigmoid transformation with gamma where INPUT is a gray scale input image

% Complete the remaining part

% transformation
output = 1 ./ (1 + exp(-1 .* gamma .* (input-0.5)));