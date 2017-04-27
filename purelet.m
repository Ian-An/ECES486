% ------------------------------------------------------------------------
% Function: PURELET for Poisson image denoising
% ------------------------------------------------------------------------
% Usage 1: XHAT = PURELET( X, LET_ID, J )
%
% Input parameters:
% X = Poisson noisy input image 
% LET_ID = LET ID; should be 0, 1, or 2. See [1].
% J = No. of Haar wavelet scales.
%  
% Output parameters:
% XHAT = Estimated image
% 
% Description: Denoises a Poisson-count noisy image using PURELET
% ------------------------------------------------------------------------
% References:
% [1] F. Luisier, C. Vonesch, T. Blu, M. Unser, "Fast Interscale Wavelet
%     Denoising of Poisson-corrupted Images", Signal Processing, vol. 90,
%     no. 2, pp. 415-427, February 2010.
% ------------------------------------------------------------------------
% Author: Sandeep Palakkal (sandeep.dion@gmail.com)
% Affiliation: Indian Institute of Technology Madras
% Created on: Feb 11, 2011
% Modified on: Feb 11, 2011
% ------------------------------------------------------------------------

function xhat = purelet( x, let_id, J )

h = [1 1];
g = [-1 1];
% [h g] = wfilters('haar');
Nor = 3;
xa = x;
delta = [];
bv_final = size( x );
for ii = 1:J
  [xw bv] = wavedec2( xa, 1, h, g );
  bv_final = [ bv(2,:) ; bv_final ];
  ln_s = prod( bv(1,:) );
  s = xw( 1:ln_s );
  xa = reshape( s, bv(1,:) );
  s = s(:);
  T2 = 6 * abs( s ) + eps;
  T2_m =  6 * abs(s-1) + eps;
  T2 = [ T2; T2; T2 ];
  T2_m = [ T2_m; T2_m; T2_m ];
  d = xw( ln_s + 1 : end );
  d = d(:);
  d_p = d + 1;
  d_m = d - 1;
  mask_0 = 1 - exp( -0.5 * d.^2 ./ T2 );
  mask_p_0 = 1 - exp( -0.5 * d_p.^2 ./ T2_m );
  mask_m_0 = 1 - exp( -0.5 * d_m.^2 ./ T2_m );
  theta_0 = [ d   mask_0 .* d ];
  theta_p_0 = [ d_p   mask_p_0 .* d_p ];
  theta_m_0 = [ d_m   mask_m_0 .* d_m ];

  if let_id == 0
    theta_all = theta_0;
    theta_m_all = theta_m_0;
    theta_p_all = theta_p_0;
  else
    d_tilde_1 = fast_cconv2(xa,[-1 0 1]');
    d_tilde_2 = fast_cconv2(xa,[-1 0 1]);
    d_tilde_3 = fast_cconv2(d_tilde_2,[-1 0 1]');
    d_tilde_1 = d_tilde_1(:);
    d_tilde_2 = d_tilde_2(:);
    d_tilde_3 = d_tilde_3(:);    
    theta_1 = [ theta_0  [d_tilde_1; d_tilde_2; d_tilde_3] ];
    theta_p_1 = [ theta_p_0 [d_tilde_1; d_tilde_2; d_tilde_3] ];
    theta_m_1 = [ theta_m_0 [d_tilde_1; d_tilde_2; d_tilde_3] ];
  end

  if let_id == 1
    theta_all = theta_1;
    theta_m_all = theta_m_1;
    theta_p_all = theta_p_1;
  elseif let_id == 2
    f = exp( -(-3:3).^2 / 2 ) / sqrt( 2*pi );
    f = f.' * f;
    p1 = fast_cconv2( reshape( abs( d_tilde_1 ), bv(1,:) ), f );
    p2 = fast_cconv2( reshape( abs( d_tilde_2 ), bv(1,:) ), f );
    p3 = fast_cconv2( reshape( abs( d_tilde_3 ), bv(1,:) ), f );    
    p = [p1(:); p2(:); p3(:)];
    theta_2 = zeros(size(d,1),6);
    theta_p_2 = zeros(size(d,1),6);
    theta_m_2 = zeros(size(d,1),6);
    for kk = 1:3
      mask_2 = 1 - exp( -1/12 * p.*p ./ abs( [s;s;s] + eps ) );
      mask_p_2 = 1 - exp( -1/12 * p.*p ./ (abs([s;s;s] - 1) + eps));
%       mask_2 = [mask_2; mask_2; mask_2];
%       mask_p_2 = [mask_p_2; mask_p_2; mask_p_2];
      theta_2(:,kk) = ( 1 - mask_2 ).* theta_1(:,kk);
      theta_2(:,kk+3) = mask_2 .* theta_1(:,kk);
      theta_p_2(:,kk) = ( 1 - mask_p_2 ).* theta_p_1(:,kk);
      theta_p_2(:,kk+3) = mask_p_2 .* theta_p_1(:,kk);
      theta_m_2(:,kk) = ( 1 - mask_p_2 ).* theta_m_1(:,kk);
      theta_m_2(:,kk+3) = mask_p_2 .* theta_m_1(:,kk);
    end
    theta_all = theta_2;
    theta_m_all = theta_m_2;
    theta_p_all = theta_p_2;
  elseif let_id > 2
    error( 'Inappropriate LET_ID' );
  end

  for jj = Nor:-1:1
    ln_jj = (jj-1)*ln_s + 1 : jj*ln_s;
    theta = theta_all( ln_jj, : );
    theta_m = theta_m_all( ln_jj, : );
    theta_p = theta_p_all( ln_jj, : );
    M = theta.' * theta;
    c = 0.5 * ( d(ln_jj).' * ( theta_m + theta_p ) + s.' * (theta_m - theta_p ) );
    c = c(:);
%     pinv(M);
    a = pinv(M) * c;
    %     a = [1 0]';
    delta = [theta * a;  delta];
  end
end
bv_final = [ bv(1,:);  bv_final ];
xhat_w = [xa(:); delta];
xhat = waverec2(xhat_w.', bv_final, h/2, fliplr(g)/2 );
end