function [R,theta,phi] = array_pattern_fft(mics,W,f,k)
% compute radiation pattern of 1 or 2 dim. microphone array
%
% mics      matrix of mic x-, y-coordinates, (z-coord. = 0)
% W         weight matrix of fixed beamformer
%           (rows are FFT coeff. for each channel)
% f         frequency in Hz (must correspond to frequency index kf of W)
% k         frequency index corresponding to f
% R         squared magnitude of rad. pattern
%           1 dim. array: R(phi)   
%           2 dim. array: R(theta,phi)   
% phi       azimuth vector in radians
% theta     elevation vector in radians (pi/2, if 1 dim. array)


  c = 343.2;
  Beta = 2*pi*f/c;                  % wave number

 
%   theta = pi/180*(0:2.5:90);
%   phi = pi/180*(-180:2.5:180);
theta = linspace(0, pi/2, 90);
phi=linspace(-pi, pi, 120);
  
  Nphi = length(phi);
  Ntheta = length(theta);
  
  U = -j*Beta*mics;
  V =[cos(phi) ; sin(phi)];  
  R = zeros(Ntheta,Nphi);
  
  for m = 1:Ntheta
     r = sin(theta(m))*V;
     n = 0.1*randn(size(mics,1),Nphi)+j*0.1*randn(size(mics,1),Nphi);
     %n = j*0.1*zeros(size(mics,1),Nphi);
     D = exp(U*r) ;               % matrix of steering vectors
     R(m,:) = abs(W(:,k)'*D).^2;
  end
  
