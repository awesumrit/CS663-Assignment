function [ output_image ] = myNearestNeighborInterpolation(im, zoom_row, zoom_col)

[m n] = size(im);
[x y] = meshgrid(1:zoom_col*n-1, 1:zoom_row*m-2);
y = floor(y/zoom_row + 1);
x = floor((x-1)/zoom_col)+1;
ind = sub2ind([m, n], y, x);
output_image = im(ind);

end
