clear all;
clc
%% Load the image
I = im2double(imread('seg_orig.tiff'));
imshow(I);

I = imnoise(I,'gaussian',0,0.3);
scale = 1e10;
I = scale*imnoise(I/scale,'poisson');
figure;imshow(I);

mesh(I);

%% Gaussian Denoise
Gaussian_Denoised_I = conv2(I, ones(3)/9, 'same');
imshow(Gaussian_Denoised_I);

%% Poisson Denoise
%Copyright (c) 2011, SANDEEP PALAKKAL
%All rights reserved.

mx = 10; % Maximum intensity of the true image
mn = 0.9; % Minimum intensity of the true image

[z im] = poisson_count( Gaussian_Denoised_I, mn, mx );

J = 5; % No. of wavelet scales
let_id = 2; %PURE-LET 0, 1, or 2.
nSpin = 5; % No. of cycle spins.
y = cspin_purelet(z,let_id,J,nSpin);

sprintf('INPUT PSNR = %f', psnr(im,z,mx) )
sprintf('OUPUT PSNR = %f', psnr(im,y,mx) )
imagesc(y);colormap(gray);
%% Convert the image back to unit8 format & Increase the histogram
K = uint8(y);
figure;K = imadjust(K);
imshow(K,[]);


%% Implement Contourlet Transform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Yue M. Lu and Minh N. Do
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	Demo_denoising.m
%	
%   First Created: 09-02-05
%	Last Revision: 07-13-09
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Denoising using the ContourletSD transform by simple hardthreholding

dfilt = 'pkva';
nlev_SD = [2 2 3 4 5];
smooth_func = @rcos;

% Pyr_mode can take the value of 1, 1.5 or 2. It specifies which one of
% the three variants of ContourletSD to use. The main differences are the
% redundancies of the transforms, which are 2.33, 1.59, or 1.33,
% respectively.

Pyr_mode = 1; 
% redundancy = 2. Set Pyr_mode = 2 for a less redundant version of the
% transform.

X = K;
X = double(X);

sigma = 30; % noise intensity

% Get the noisy image
Xn = X;

% load the pre-computed norm scaling factor for each subband. For images
% of sizes other than 512 * 512, run ContourletSDmm_ne.m again.
eval(['load SDmm_' num2str(size(Xn,1)) '_' num2str(Pyr_mode)]);

% Take the ContourletSD transform
Y = ContourletSDDec(Xn, nlev_SD, Pyr_mode, smooth_func, dfilt);

% The redundancy of the transform can be verified as follows.
dstr1 = whos('Y');
dstr2 = whos('Xn');
dstr1.bytes / dstr2.bytes

% Apply hard thresholding on coefficients
Yth = Y;
for m = 2:length(Y)
  thresh = 3*sigma + sigma * (m == length(Y));
  for k = 1:length(Y{m})
    Yth{m}{k} = Y{m}{k}.* (abs(Y{m}{k}) > thresh*E{m}{k});
  end
end

% ContourletSD reconstruction
Xd = ContourletSDRec(Yth, Pyr_mode, smooth_func, dfilt);

L = imadjust(Xd/255);
figure;imagesc(L);colormap(gray);
imshow(L);
title(['Denoising using the contourletSD transform. ']);


%% Smooth Image

background = imopen(L,strel('disk',15));
%Apply guassian smoothing filters
smoothed_background = imgaussfilt(background,8);
%Get the foreground
I2 = L - smoothed_background;
bw = imbinarize(I2);
bw = bwareaopen(bw, 50);
bw = imfill(bw,'holes');
figure;imagesc(bw);colormap(gray);
%Generate the distance map
D = bwdist(bw);
figure;imagesc(D);
%Implement watershed segmentation algorithm
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(bw), hy, 'replicate');
Ix = imfilter(double(bw), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);%gradient magnitude
figure;imagesc(gradmag);colormap(gray);
imshow(gradmag)
C =~bw;
D = -bwdist(C);
D(C) = -Inf;
L = watershed(D);% watershed image
figure;imagesc(L);
Wi = label2rgb(L);
%Gnerate scatter plot
s = regionprops(L, 'centroid');
centroids = cat(1, s.Centroid);
size(centroids)
imshow(Wi)
hold(imgca,'on')
plot(imgca,centroids(:,1), centroids(:,2),'*r')
title(['Segmentation of original image ']);
hold(imgca,'off')

%Generate perimeter
perimeter = regionprops('table',bw,'Perimeter')
%Generate area
area = regionprops('table',bw,'EquivDiameter')

