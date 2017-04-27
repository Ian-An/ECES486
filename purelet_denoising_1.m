% Poisson image denoising using PURE-LET
% ------------------------------------------------------------------------
% References:
% [1] F. Luisier, C. Vonesch, T. Blu, M. Unser, "Fast Interscale Wavelet
%     Denoising of Poisson-corrupted Images", Signal Processing, vol. 90,
%     no. 2, pp. 415-427, February 2010.
% ------------------------------------------------------------------------
% Author: Sandeep Palakkal (sandeep.dion@gmail.com)
% Affiliation: Indian Institute of Technology Madras
% Created on: Feb 11, 2011
% ------------------------------------------------------------------------

clear


x = double(imread('test.tiff'));

mx = 10; % Maximum intensity of the true image
mn = 0.9; % Minimum intensity of the true image

[z im] = poisson_count( x, mn, mx );

J = 5; % No. of wavelet scales
let_id = 2; %PURE-LET 0, 1, or 2.
nSpin = 5; % No. of cycle spins.
y = cspin_purelet(z,let_id,J,nSpin);

sprintf('INPUT PSNR = %f', psnr(im,z,mx) )
sprintf('OUPUT PSNR = %f', psnr(im,y,mx) )

figure, imshow(z,[])
figure, imshow(y,[])