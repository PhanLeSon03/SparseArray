function [theta,BS] = Directivity(mics,W,f)

    vs = 343.2;
    theta = pi/180*(-90:2.5:90);
    phi = 0;

    Ntheta = length(theta);

    
    
    V =[cos(phi) ; sin(phi)];     
    BS = zeros(length(f),Ntheta);
    
    for m = 1:length(f)
        k = 2*pi*f(m)/vs;
        U = j*k*mics;
        for iTheta = 1:Ntheta
           er = sin(theta(iTheta))*V;
           D = exp(U*er);               % matrix of steering vectors
           BS(m,iTheta) = abs(W(:,m)'*D).^2;
        end
   end