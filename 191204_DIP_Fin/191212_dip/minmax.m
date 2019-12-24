function minmax = minmax(im)


maxval = max(max(im));
minval = min(min(im));
denom = maxval-minval;
vector = im - minval*ones(128,128);
minmax = vector./denom;
end
