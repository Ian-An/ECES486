clear all;
clc
%% Load the image
I = im2double(imread('seg_orig.tiff'));
I = imnoise(I,'gaussian',0,0.3);
scale = 1e10;
I = scale*imnoise(I/scale,'poisson');
imshow(I);


%% Michel's method
h = fspecial('gaussian');
imfilt = imgaussfilt(I,0.9);
for j=1:10000
     imfilt = imfilter(imfilt, h, 'symmetric');
end
imhfreq = max((I - imfilt), zeros(size(I)));
imf = imadjust(imhfreq);
for k =1:10
    imf = medfilt2(imf,[3 3]);
end
imf = mat2gray(imf);
a = imadjust(imf)
imshow(a)

background = imopen(imf,strel('disk',15));
%Apply guassian smoothing filters
smoothed_background = imgaussfilt(background,8);
%Get the foreground
I2 = imf - smoothed_background;
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


