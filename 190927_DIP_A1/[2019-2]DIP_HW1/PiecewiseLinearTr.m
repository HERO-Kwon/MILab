function output = PiecewiseLinearTr(input,a,b) %
% PiecewiseLinearTr(IM,A,B) applies a piecewise linear transformation to the pixel values
% of the input image INPUT, where A and B are vectors containing the x and y coordinates
% of the ends of the line segments. INPUT can be of type DOUBLE,
% and the values in A and B must be between 0 and 1 (normalized intensity values). %
% For example:
%
% PiecewiseLinearTr(x,[0,1],[1,0])
%
% simply do negative transform inverting the pixel values.
%

if length(a) ~= length (b)
    error('Vectors A and B must be of equal size');
end

% set output size equal to input size
output= zeros(size(input));
% iterate for every vector
for i = 1:length(a)-1
    % set vector
    a1 = a(i);
    a2 = a(i+1);
    b1 = b(i);
    b2 = b(i+1);
    
    % mask image
    filter = (input >= a1) & (input <= a2);
    
    % transformation
    % Equation: s = ((b2-b1)/(a2-a1))(r-a1) + b1
    m = (b2-b1)/(a2-a1);
    % merge output
    output = output + filter.*(m*(input-a1) + b1);
end
    
% Complete the remaining part
