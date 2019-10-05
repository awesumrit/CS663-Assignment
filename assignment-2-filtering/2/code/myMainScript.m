%% MyMainScript

tic;
%% Your code here
str_struct = load('../data/barbara.mat');
input_one = double(str_struct.imageOrig);
input_two = double(imread('../data/grass.png'));
input_three = double(imread('../data/honeyCombReal.png'));

sig_s1 = 20;
% sig_s1 = 0.9 * 20;
% sig_s1 = 1.1 * 20;
sig_s2 = 18;
% sig_s2 = 0.9 * 18;
% sig_s2 = 1.1 * 18;
sig_s3 = 18;
% sig_s3 = 0.9 * 18;
% sig_s3 = 1.1 * 18;

sig_r1 = 9;
% sig_r1 = 0.9 * 9;
% sig_r1 = 1.1 * 9;
sig_r2 = 27.5;
% sig_r2 = 0.9 * 27.5;
% sig_r2 = 1.1 * 27.5;
sig_r3 = 33.275;
% sig_r3 = 0.9 * 33.275;
% sig_r3 = 1.1 * 33.275;

window_1 = 5;
window_2 = 3;
window_3 = 3;

[corrupted_one, mask1, output_one, rmsd1, sig1] = myBilateralFiltering(input_one, sig_s1, sig_r1, window_1);
[corrupted_two, mask2, output_two, rmsd2, sig2] = myBilateralFiltering(input_two, sig_s2, sig_r2, window_2);
[corrupted_three, mask3, output_three, rmsd3, sig3] = myBilateralFiltering(input_three, sig_s3, sig_r3, window_3);



figure();
subplot(1, 3, 1)
imshow(uint8(input_one), [], 'InitialMagnification', 'fit');
title('Barbara (Original)');
axis on,colorbar
subplot(1, 3, 2)
imshow(uint8(corrupted_one), [], 'InitialMagnification', 'fit');
title('Barbara (Corrupted)');
axis on,colorbar
subplot(1, 3, 3)
imshow(uint8(output_one), [], 'InitialMagnification', 'fit');
title('Barbara (Filtered)');
axis on,colorbar
figure();
subplot(1, 2, 1)
[x, y] = meshgrid(-50:1:50, -50:1:50);
z = exp(-(x.^2 + y.^2)./(2*(sig_s1^2)))/(sqrt(2*pi*(sig_s1^2)));
surf(x,y,z)
axis on, colorbar
subplot(1, 2, 2)
z = z / max(max(z));
imshow(z)
colormap('jet')
axis on,colorbar

figure();
subplot(1, 3, 1)
imshow(uint8(input_two), [], 'InitialMagnification', 'fit');
title('Grass (Original)');
axis on,colorbar
subplot(1, 3, 2)
imshow(uint8(corrupted_two), [], 'InitialMagnification', 'fit');
title('Grass (Corrupted)');
axis on,colorbar
subplot(1, 3, 3)
imshow(uint8(output_two), [], 'InitialMagnification', 'fit');
title('Grass (Filtered)');
axis on,colorbar
figure();
subplot(1, 2, 1)
[x, y] = meshgrid(-50:1:50, -50:1:50);
z = exp(-(x.^2 + y.^2)./(2*(sig_s2^2)))/(sqrt(2*pi*(sig_s2^2)));
surf(x,y,z)
axis on, colorbar
subplot(1, 2, 2)
z = z / max(max(z));
imshow(z)
colormap('jet')
axis on,colorbar

figure();
subplot(1, 3, 1)
imshow(input_three, [], 'InitialMagnification', 'fit');
title('HoneyComb (Original)');
axis on,colorbar
subplot(1, 3, 2)
imshow(corrupted_three, [], 'InitialMagnification', 'fit');
title('HoneyComb (Corrupted)');
axis on,colorbar
subplot(1, 3, 3)
imshow(output_three, [], 'InitialMagnification', 'fit');
title('HoneyComb (Filtered)');
axis on,colorbar
figure();
subplot(1, 2, 1)
[x, y] = meshgrid(-50:1:50, -50:1:50);
z = exp(-(x.^2 + y.^2)./(2*(sig_s3^2)))/(sqrt(2*pi*(sig_s3^2)));
surf(x,y,z)
axis on, colorbar
subplot(1, 2, 2)
z = z / max(max(z));
imshow(z)
colormap('jet')
axis on,colorbar

toc;