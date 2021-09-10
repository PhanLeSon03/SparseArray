
function W = bf_coefs(mics,theta_d,phi_d,resp,f,mue, null, d_mics,vs)
%
% compute fixed beamformer weight in frequency domain 
%
% mics     (x,y) coordinates of array (see mics_config.m)
% theta_d  elevations in deg. of desired direction followed by direction of nulls
% phi_d    azimuths in deg., of desired direction followed by direction of nulls 
% resp     beamformer response in desired direction followed by direction of nulls
% f        frequency vector in Hz at which W is computed
% mue      regularization parameter
% null     using spatial null or not
% W        N x Nhigh-Nlow+1 array of beamformer weights


    %vs = 340;

    % compute matrices used for optimzation

    theta_d = theta_d.* pi / 180;
    phi_d = phi_d.* pi / 180;

    [N,K] = size(mics);
    if (K < 2) || (N < 1)
       error('bad microphone positions');
    end

    if K == 2                      % 2 dim. array 
       mics(:,1) = mics(:,1);  
       mics(:,2) = mics(:,2); 
       rn = [mics zeros(N,1)];
    else
       rn = mics;
    end

    Ntheta = length(theta_d);
    if (Ntheta ~= length(phi_d))
       error('angle vectors must have same sizes');
    end

    er = [sin(theta_d).*cos(phi_d) ; sin(theta_d).*sin(phi_d) ; cos(theta_d)];  % steering vector
    Rc = rn*er;            % used to compute matrix C = exp(j*beta*rn*er)

    % compute distance matrix of all microphones (used to compute correlation matrix)

    x = rn(:,1);
    x = x(:,ones(N,1));
    dx = x - x.';
    y = rn(:,2);
    y = y(:,ones(N,1));
    dy = y - y.';
    if K == 2
       dR = sqrt(dx.^2 + dy.^2);
    else
       z = rn(:,3);
       z = z(:,ones(N,1));
       dz = z - z.';
       dR = sqrt(dx.^2 + dy.^2 + dz.^2);
    end

% perform optimization for each frequency to find reference values

nf = length(f);
W = zeros(N,nf);

step = 0.005;  % step of searching mue

for l = 1:nf

   if null==1
       beta = 2*pi*f(l)/vs;     % wave number
       C = exp(j*beta*Rc);      % constraints matrix
       A = sinc(beta/pi*dR);    % microphone correlation matrix
                                % (isotropic noise field, spherical mic radiation pattern)
       Amue = A + mue*eye(size(A));   % regularization
       B = (Amue^-1)*C;
       Lambda = (C'*(-B))\resp(:); % Lagrange multiplicator
       W(:,l) = B*Lambda;       % optimum coefficient vector at given frequency
   else
       beta = 2*pi*f(l)/vs;     % wave number
       C = exp(j*beta*Rc);      % constraints matrix
       A = sinc(beta/pi*dR);    % microphone correlation matrix
                                % (isotropic noise field, spherical mic radiation pattern)

       Amue = A + mue*eye(size(A));   % regularization
       B = (Amue^-1)*C(:,1);
       W(:,l) = B/(C(:,1)'*B);  

   end
end 


% White Noise Gain computation 
function out = computeWNG(weight)
    den = real(sum(weight.*conj(weight)));
    gain =  1*1/(den);
    out = 10*log10(gain);
