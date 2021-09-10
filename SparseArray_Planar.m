clear
clc
close all;

c = 340;
ft = 16000;
NFIR = 128;
fStep = (ft/2)/(NFIR/2+1);
fLow = 1000;
fHigh = 4500;
kTUH = floor(fLow/fStep);
kTOH = floor(fHigh/fStep);
f = fStep*(kTUH:kTOH);       % frequencies used to compute W
fp = [1080 2000 4000];       % list frequencies are ploted
klow = kTUH;                 % low index
kup = kTOH;                  % high index
kf = round(fp/fStep);
fp = fStep*kf;               % frequencies rounded to FFT grid
kf = kf-klow+1;              % index used in W matrix corresponding to fp 

% Uniform array
N = 101;
dH = 0.01;
gamma = 0.8;                 % factor of engergy threshold 
numCluster = 15;             % number of group
numPrsnt = 4;                % number of presentative sensors in each group 

n1 = -(N-1)/2:1:(N-1)/2;
n2 = n1;
[n1,n2] = meshgrid(n1,n2);
aux=sqrt(n1.^2+n2.^2);
d=dH;
thetaC = pi/8;
thetaC1 = pi/16;
W_2D = zeros(N, N,kTOH-kTUH+1);

dBmax = 40;
theta = linspace(0, pi/2, 90);
phi=linspace(-pi, pi, 120);
Ntheta = length(theta);
Nphi = length(phi);
B_REF = zeros(Ntheta,Nphi);
x = linspace(0,2.2,Ntheta);
y =sin(pi*x)./(pi*x);
y(1) = 1;

B_REF(:,1) = abs(y);
for i=2:120
    B_REF(:,i) = B_REF(:,1);
end

figure();
plot(B_REF(:,1));
ylabel('Gain');
xlabel('Azimuth angle');
set(gcf,'color','w');


BP = B_REF;
figure()
% 3D plot
B_REF = max(0,20*log10(B_REF+eps)+dBmax);
Xc = B_REF .* (sin(theta')*cos(phi));
Yc = B_REF .* (sin(theta')*sin(phi));
Zc = B_REF .* (cos(theta')*ones(1,length(phi)));
set(gcf,'defaultAxesFontSize',12);
mesh(Xc,Yc,Zc,B_REF);
title('Reference beam pattern: dB');
set(gcf,'color','w');
alph = 2*max(x)/pi;   
figure()
%%
%calculate the weight for broadband beamformer ============================
for k=kTUH:kTOH

    Rc = k*fStep*N*dH/c;
    boudary  = find(aux> Rc );
    temp = aux/Rc;    
    temp1 = alph*pi*asin(temp);
    temp2 = abs(sin(temp1)./temp1);

    temp2(round(N/2),round(N/2))=1;
    temp2(boudary)=0;
    Hd = (temp2);

    mesh(n1,n2,Hd)
    axis off
    h =fftshift(ifft2(rot90(fftshift(rot90(Hd,2)),2)));

    W_2D(:,:,k-kTUH+1) = h;

    pause(0.05);
end

W = ones(N*N,kTOH-kTUH+1);
for yPos=1:N
   for xPos=1:N
      W((yPos-1)*N+xPos,:) = (W_2D(yPos,xPos,:));
   end
end

%verify the beam pattern ==================================================

% virtual uniform array
mics_ref = zeros(N*N,2);
for yPos = 1: N
  for xPos = 1:N
      mics_ref(xPos + N*(yPos-1), 1) = (xPos)*dH-round(N/2)*dH;   
      mics_ref(xPos + N*(yPos-1), 2) = (yPos)*dH-round(N/2)*dH;
  end
end

%%%%%%%%%% plot beam response
pos = [0.045 0.045 0.45 0.45];
for k = 1:length(fp)
    figure('numbertitle','off','name','Fixed Array radiation pattern (dB)',...
                  'Units','normal','Position',pos);

    %Plotting beam-pattern for 1st looking direction 
    [R,t,p] = array_pattern_fft(mics_ref,W,fp(k),kf(k)); 
    R = R/max(R(:));
    RdB = max(0,10*log10(R+eps)+dBmax);
 
    % 3D plot
    Xc = RdB .* (sin(t')*cos(p));
    Yc = RdB .* (sin(t')*sin(p));
    Zc = RdB .* (cos(t')*ones(1,length(p)));

    mesh(Xc,Yc,Zc,RdB);
       
    pos(1) = pos(1) + 0.3;
    
    ght = title(sprintf('f = %3.2f Hz, unit dB',fp(k)));
    colormap jet;
    set(gcf,'color','w');
end   

[goc, BS] = Directivity(mics_ref,W,f);
BS_dB = max(0,20*log10(BS+eps)+dBmax);
[Fxx,Fyy] = meshgrid(goc/pi*180,f);

figure('numbertitle','off','name','Fixed Array radiation pattern (dB)',...
                  'Units','normal','Position',[0.1 0.1 0.5 0.5]);
surf(Fxx,Fyy,sqrt(BS));
xlabel('Azimuth');
ylabel('Hz');
zlabel('Magnitude');
axis tight
set(gcf,'color','w');
grid on
box on
% Sparse Array Design
pos = [0.045 0.045 0.45 0.45];
pos(1) = pos(1) + 0.2; 
figure('numbertitle','off','name','weight spectrum of sensor array',...
      'Units','normal','Position',pos);
  
imagesc(f,[],abs(W));
ylabel('sensor index');
xlabel('Hz');
colorbar
colormap jet;
set(gcf,'color','w');

X = bsxfun(@minus,W,mean(W));
% Do the PCA
[coeff,score,latent] = pca(X);

% Calculate eigenvalues and eigenvectors of the covariance matrix
[V,E] = eig(W'*W);

figure();
imagesc(abs(score(:,1:5)))
ylabel('sensor index');
xlabel('Dimentional reduction of the frequencies');
colorbar
set(gcf,'color','w');
colormap jet;

opts = statset('Display','final');
for i=1:N*N
  e(i) = norm(score(i,1:5));
end

maxScore = max(abs(e));
idxScore = find(e>gamma*maxScore);
idxScoreCom = find(e<=gamma*maxScore);

[idxMic, Cx, sumd, Dx] = kmeans(abs(score(idxScoreCom,1:5)),numCluster,'Replicates',500,'Options',opts);
sum(sumd)
x_sparse = zeros(N*N,2);
idxSpare = idxScore;
x_sparse(idxScore,:) = mics_ref(idxScore,:);
for iCluster= 1:numCluster
    idxC = find(idxMic==iCluster);
    DisArray = Dx(idxC,iCluster);
    [closetMic,idxMin] = mink(DisArray,numPrsnt);
    x_sparse(idxScoreCom(idxC(idxMin)),:) = mics_ref(idxScoreCom(idxC(idxMin)),:);
    idxSpare = [idxSpare, idxScoreCom(idxC(idxMin))];
end

x_sparse = x_sparse(idxSpare,:);
SUA_n = ceil(sqrt(length(x_sparse))) + 2;
% small size uniform array
x_uniform_1 = zeros(SUA_n*SUA_n,2);
for yPos = 1: SUA_n
  for xPos = 1:SUA_n
      x_uniform_1(xPos + SUA_n*(yPos-1), 1) = (xPos)*dH-round(SUA_n/2)*dH;   
      x_uniform_1(xPos + SUA_n*(yPos-1), 2) = (yPos)*dH-round(SUA_n/2)*dH;
  end
end
% big size uniform array
x_uniform_2 = zeros(SUA_n*SUA_n,2);
for yPos = 1: SUA_n
  for xPos = 1:SUA_n
      x_uniform_2(xPos + SUA_n*(yPos-1), 1) = (xPos)*dH*N/(SUA_n-1)-round(SUA_n/2)*dH*N/(SUA_n-1);   
      x_uniform_2(xPos + SUA_n*(yPos-1), 2) = (yPos)*dH*N/(SUA_n-1)-round(SUA_n/2)*dH*N/(SUA_n-1);
  end
end


pos = [0.5 0.5 0.4 0.4];
myFig = figure('numbertitle','off','name','sparse array layout (cm)','Units','normal',...
   'Position',pos);%,'Menubar','none'
plot(100*x_sparse(:,1),100*x_sparse(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','r',...
 'MarkerSize',6);
hold on
plot(100*x_uniform_1(:,1),100*x_uniform_1(:,2),'x','MarkerEdgeColor','g','MarkerFaceColor','g',...
 'MarkerSize',6);
plot(100*x_uniform_2(:,1),100*x_uniform_2(:,2),'*','MarkerEdgeColor','b','MarkerFaceColor','b',...
 'MarkerSize',6);
 
title('Planar arrays in cm'); 
grid on
set(gcf,'color','w');
legend('sparse array', 'small size uniform array', 'big size uniform array');
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');

%%
Distance_SA = zeros(length(x_sparse),length(x_sparse));
Distance_SUA = zeros(length(x_uniform_1),length(x_uniform_1));
Distance_BUA = zeros(length(x_uniform_2),length(x_uniform_2));
for i=1:length(x_sparse)
   Distance_SA(i,:) =  sqrt((x_sparse(i,1)-x_sparse(:,1)).^2+(x_sparse(i,2)-x_sparse(:,2)).^2);
end
for i=1:length(x_uniform_1)
   Distance_SUA(i,:) =  sqrt((x_uniform_1(i,1)-x_uniform_1(:,1)).^2+(x_uniform_1(i,2)-x_uniform_1(:,2)).^2);
   Distance_BUA(i,:) =  sqrt((x_uniform_2(i,1)-x_uniform_2(:,1)).^2+(x_uniform_2(i,2)-x_uniform_2(:,2)).^2);
end

Distance ={Distance_SA,...
        Distance_SUA,...
        Distance_BUA};
    
array_config ={x_sparse,...
              x_uniform_1,...
              x_uniform_2};
Plot_Title = {'(a) new sparse array', '(b) small size uniform array', '(c) big size uniform array', '(d) incoherent design'};
Plot_Color = {'r', 'g', 'b', 'k'};
Marker = {
'*' ,... %Asterisk
'x' ,... %Cross
'^' ,... %Upward-pointing triangle
'v' ,... %Downward-pointing triangle
'>' ,... %Right-pointing triangle
'<' ,... %Left-pointing triangle   
'square' ,... %or 's'   Square
'diamond' ,... %or 'd'  Diamond
'o' ,... %Circle
'pentagram' ,... %or 'p'  Five-pointed star (pentagram)
'hexagram' ,... %or 'h'''  Six-pointed star (hexagram)
'none',...  %No marker (default)
'+',... %  Plus sign
'.'  %Point
};
for iConfig=1:3
    x_opt = array_config{iConfig};
    Nf = length(f);
    Nsparse = length(x_opt);
    h_map = zeros(Nsparse,Nf);
    DP = zeros(Nphi,Nf);
    idxTheta_M = 1:30;
    idxTheta_S = 31:Ntheta ;
    V =[cos(phi) ; sin(phi)];  
    BP_M = BP(idxTheta_M,:);
    BP_MV = BP_M(:);
    BP_S=BP(idxTheta_S,:);
    BP_SV = BP_S(:);
    C1= ones(length(f),1)*0.0002*length(idxTheta_M)*Nphi;
    C2= ones(length(f),1)*0.0006*length(idxTheta_S)*Nphi;

    C2(26:end)= 0.0006*length(idxTheta_S)*Nphi;
    WNG = zeros(length(f),1);
    DF = zeros(length(f),1);
    BPE = zeros(length(f),1);
    for iF = 1:length(f)

        beta = 2*pi*f(iF)/c;             % wave number
        D = zeros(Ntheta, Nphi, Nsparse);

        for m = 1:Ntheta
            r = sin(theta(m))*V;      
            D(m,:,:) = exp(1j*beta*x_opt*r)' ;               % matrix of steering vectors        
        end
        D_M = D(idxTheta_M,:,:); % main lope
        D_S = D(idxTheta_S,:,:); % side lope
        d = squeeze(D(1,1,:))  ;  % looking direction    

        D_MV = reshape(D_M,size(D_M,1)*size(D_M,2),size(D_M,3),[]);
        D_SV = reshape(D_S,size(D_S,1)*size(D_S,2),size(D_S,3),[]);

        cvx_begin 
              variable xh(Nsparse) complex 
              minimize( norm(xh))
              subject to 
                   d'*xh == 1;                         % looking direction constraint
                   norm(BP_MV - D_MV*conj(xh)) <= C1(iF);  % main lope constraint
                   norm(BP_SV - D_SV*conj(xh)) <= C2(iF); % side lope constraint
        cvx_end;

        h=xh;
        h_map(:,iF) = h;
        WNG(iF) = h'*h;
       Shi = (sin(beta*Distance{iConfig})./(beta*Distance{iConfig}));
       Shi(logical(eye(size(Shi)))) = 1;
       DF(iF) = xh'*Shi*xh;
       
       if norm(h) < inf
           BPE(iF) = (sum(abs((BP_MV) - (D_MV*conj(h)))) + ...
           sum(abs((BP_SV) - (D_SV*conj(h)))))/Ntheta/Nphi;
       else
           BPE(iF) = inf;
       end
    end
    S.(sprintf('h_map%d', iConfig)) = h_map;
    S.(sprintf('WNG%d', iConfig))= WNG;
    %directivity factor
    S.(sprintf('DF%d', iConfig))= DF;
    %Beampattern error everage
    S.(sprintf('BPE%d', iConfig))= BPE;
end


%%
Plot_Title1 = {'(d) new sparse array', '(e) small size uniform array', '(f) big size uniform array', '(d) incoherent design'};
set(gcf,'defaultAxesFontSize',12);

pos = [0.045 0.045 0.45 0.45];
myFig =    figure('numbertitle','off','name','Array radiation pattern (dB)',...
                  'Units','normal','Position',pos);
    
for iConfig=1:3
    x_opt = array_config{iConfig};
    k = 2; 
    subplot(3,1,iConfig);
    %Plotting beam-pattern for 1st looking direction 
    [R,t,p] = array_pattern_fft(x_opt,S.(sprintf('h_map%d', iConfig)),fp(k),kf(k));
    
    diff = sum(sum(abs(sqrt(R) - BP)));
    Error = diff/size(BP,1)/size(BP,2)
    R = R/max(R(:));

    RdB = max(0,10*log10(R+eps)+dBmax);

   
    % 3D plot
    Xc = RdB .* (sin(t')*cos(p));
    Yc = RdB .* (sin(t')*sin(p));
    Zc = RdB .* (cos(t')*ones(1,length(p)));
 
    mesh(Xc,Yc,Zc,RdB);
    axis([-dBmax dBmax -dBmax dBmax 0 dBmax]);   
    
        hold on;
    plot3(dBmax*cos(p),dBmax*sin(p),zeros(length(p),1),'b--');
    plot3(dBmax*sin(t),zeros(length(t),1),dBmax*cos(t),'b--');
    plot3(dBmax*sin(-t),zeros(length(t),1),dBmax*cos(-t),'b--');
    plot3(zeros(length(t),1),dBmax*sin(t),dBmax*cos(t),'b--');
    plot3(zeros(length(t),1),dBmax*sin(-t),dBmax*cos(-t),'b--');
    hold off;
    
    pos(1) = pos(1) + 0.3;
    xlabel('dB');
    ylabel('dB');
    zlabel('dB');
    title(Plot_Title(iConfig));
    colormap jet;
    grid on
    box on
  
end  
set(gcf,'color','w');
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');
%plotting cross-section over frequencies
pos = [0.045 0.045 0.45 0.45];
myFig =    figure('numbertitle','off','name','White noise gain','Units','normal',...
       'Position',pos);
hold on
for iConfig=1:3
    plot(f,10*log10(1./S.(sprintf('WNG%d', iConfig))),strcat('-',Plot_Color{iConfig},Marker{iConfig}),'MarkerEdgeColor',Plot_Color{iConfig});
    xlabel('Hz');
    ylabel('WNG (dB)');
    legend('sparse array', 'small size uniform array', 'big size uniform array');
    axis tight
    set(gcf,'color','w');
    set(gca,'FontSize', 12);
    grid on
    box on
end
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal')

pos(1) = pos(1) +0.1;
% Plot DF
myFig =   figure('numbertitle','off','name','Directivity factor','Units','normal',...
       'Position',pos);
hold on
for iConfig=1:3
    %plot(f,10*log10(1./S.(sprintf('DF%d', iConfig))),Plot_Color{iConfig});
    plot(f,10*log10(1./S.(sprintf('DF%d', iConfig))),strcat('-',Plot_Color{iConfig},Marker{iConfig}),'MarkerEdgeColor',Plot_Color{iConfig});
    xlabel('Hz');
    ylabel('DF (dB)');
    legend('sparse array', 'small size uniform array', 'big size uniform array');
    axis tight
    ylim([0 12])
    set(gcf,'color','w');
    set(gca,'FontSize', 12);
    grid on
    box on
end
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');

% Plot beampattern error everage
pos(1) = pos(1) +0.1;
myFig =   figure('numbertitle','off','name','Beamp pattern error everage','Units','normal',...
       'Position',pos);
hold on
for iConfig=1:3
    plot(f,S.(sprintf('BPE%d', iConfig)),strcat('-',Plot_Color{iConfig},Marker{iConfig}),'MarkerEdgeColor',Plot_Color{iConfig});
    xlabel('Hz');
    ylabel('BPE');
    legend('new sparse array', 'small size uniform array', 'big size uniform array');
    axis tight
    ylim([0 0.05])
    set(gca,'FontSize', 12);
    set(gcf,'color','w');
    grid on
    box on
end
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');

myFig =  figure('numbertitle','off','name','Fixed Array radiation pattern (dB)',...
                      'Units','normal','Position',[0.1 0.1 0.5 0.5]);
for iConfig=1:3
    x_opt = array_config{iConfig};
    [goc, BS] = Directivity(x_opt,S.(sprintf('h_map%d', iConfig)),f);
    BS_dB = max(0,10*log10(BS)+dBmax);
    [Fxx,Fyy] = meshgrid(goc/pi*180,f);  
    subplot(3,1,iConfig);
    surf(Fxx,Fyy,sqrt(BS)); 
    title(Plot_Title1(iConfig));
    xlabel('Azimuth');
    ylabel('Hz');
    zlabel('Magnitude');
    ylim([1000,4500]);
    zlim([0 1]);
    hold on
    set(gcf,'color','w');
    set(gca,'FontSize', 12);
    grid on
    box on
end
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');
