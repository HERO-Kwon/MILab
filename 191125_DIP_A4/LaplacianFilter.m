function output = LaplacianFilter(I,M)
size_I = size(I);
filtered = zeros(size_I);
% for color image
if length(size_I) == 3
    ch = size_I(3);
    % apply Laplacian filter
    for i=1:ch
        filtered(:,:,i) = conv2(I(:,:,i),M,'same');
    end
else % for grayscale image
    filtered = conv2(I,M,'same');
end

output = filtered;
end

