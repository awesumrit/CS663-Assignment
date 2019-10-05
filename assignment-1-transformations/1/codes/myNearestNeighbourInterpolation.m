function [ output_image ] = myNearestNeighbourInterpolation(input, zoom_row, zoom_col)

[im, map] = imread(input); 
[row, col] = size(im); 

zr=zoom_row*row - (zoom_row - 1);
zc=zoom_col*col - (zoom_col - 1);

for i = 1:zr
    x=i/zoom_row;
    mapi=round(x); %calculating the nearest pixel around ith pixel 
    if mapi==0 
        mapi=1; 
    end
    for j=1:zc
        y=j/zoom_col;
        mapj=round(y);
        if mapj==0 
            mapj=1;
        end
        output_image(i,j)=im(mapi,mapj); %zooming to nearest neighboour 
        end 
end 