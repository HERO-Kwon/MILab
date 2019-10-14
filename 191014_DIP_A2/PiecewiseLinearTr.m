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

% Complete the remaining part
