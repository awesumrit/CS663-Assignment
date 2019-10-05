function [ output_image ]  = myBilinearInterpolation(im,zoom_row,zoom_col)


[row, col] = size(im); 
%zooming factor
zr=zoom_row*row - (zoom_row - 1);
zc=zoom_col*col - (zoom_col - 1);

for i=1:zr
    i_index = i -1;
    x=i_index/zoom_row;
    
    x1=floor(x);
    x2=ceil(x);

    x1 = x1 +1;
    x2 = x2 +1;
    xint=rem(x,1);
    for j=1:zc
        j_index = j -1;
        y=j_index/zoom_col;
        
        y1=floor(y);
        y2=ceil(y);

        yint=rem(y,1);
        y1 = y1 +1;
        y2 = y2 +1;
        
        if (x2 <= row & x1 <= row & y1 <= col & y2 <= col)

            bot_left = im(x1,y1);
            top_left = im(x1,y2);
            bot_right = im(x2,y1);
            top_right = im(x2,y2);
        
            output_image(i,j) = bot_right*xint*(1-yint) + bot_left*(1-xint)*(1-yint) + top_right*xint*yint + top_left*(1-xint)*yint ;
        end
    end
end