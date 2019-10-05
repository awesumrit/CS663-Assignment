function [corrupted_img, mask, output_image, rmsd, sig]  = myBilateralFiltering(img, sig_s, sig_r, window)

[r, c] = size(img);
% max_int = max(max(img));
img_scaled = img;
% ./max_int;
sig = double(max(max(img_scaled)) - min(min(img_scaled)))/20;
mask = sig * randn(r, r, 1);
corrupted_img = img_scaled + mask;
output_image = zeros(r, c);
w = round((window-1)/2);
corrupted_img1 = padarray(corrupted_img, [w, w], 'replicate');
h = waitbar(0,'Applying bilateral filter...');
set(h,'Name','Bilateral Filter Progress');
for i = w+1:w+r
    for j = w+1:w+c
        k = i-w:i+w;
        l = j-w:j+w;
        f_r = (exp(-((corrupted_img1(k, l) - corrupted_img1(i, j)).^2)./(2*(sig_r)^2)))./sqrt(2*pi*sig_r^2);
        [k_r, l_c] = find(corrupted_img1(k, l));
        g_s = reshape(exp(-((k_r).^2 + (l_c).^2)./(2*(sig_s)^2))./sqrt(2*pi*sig_s^2), size(f_r));
        w_p = sum(f_r .*g_s, 'all');
        I_i = corrupted_img1(k, l) .* f_r .* g_s;
        output_image(i-(w), j-(w)) = (sum(I_i, 'all')/w_p);
    end
waitbar(i/r);
end
close(h);
 
rmsd = sqrt(sum((img_scaled - output_image).^2, 'all')/(r*c));
disp(rmsd);

end
