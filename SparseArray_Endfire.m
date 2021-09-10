clear;
clc;
close all;

c = 340;                     % speed of sound in m/s
Nfft = 128;                  % FFT length used to implement FFT filterbank
Nfh = Nfft/2+1;              % number of frequency points in [0,Fs/2]
Fs = 16000;
Fsh = Fs/2;
fl = 500;                    % lower and upper cutoff frequencies of filterbank
fu = 6000;
klow = round(fl/Fsh*Nfh);    % lower index
kup = round(fu/Fsh*Nfh);     % upper index
f = Fsh/Nfh*(klow:kup)';     % frequencies used to compute T and W

N = 141;                     % number of microphone
dH = 0.01;                   % minimum sensor distance in m
n = -(N-1)/2:(N-1)/2;
x_array = n*dH;              % sensor positions

gamma = 0.5;                 % factor of engergy threshold 
numCluster = 7;              % number of groups
numPrsnt = 1;                % number of presentative sensors in each group


phi = linspace(0, pi, 180);  % discrete Theta angle 
Nphi = length(phi);


Plot_Title = {'(a) sparse array', '(b) small size uniform array', '(c) big size uniform array', '(d) coherent design'};
Plot_Color = {'r', 'g', 'b', 'k'};
Marker = {
'*' ,...         %Asterisk
'x' ,...         %Cross
'^' ,...         %Upward-pointing triangle
'v' ,...         %Downward-pointing triangle
'>' ,...         %Right-pointing triangle
'<' ,...         %Left-pointing triangle   
'square' ,...    %or 's'   Square
'diamond' ,...   %or 'd'  Diamond
'o' ,...         %Circle
'pentagram' ,... %or 'p'  Five-pointed star (pentagram)
'hexagram' ,...  %or 'h'''  Six-pointed star (hexagram)
'none',...       %No marker (default)
'+',...          %Plus sign
'.'              %Point
};

BP = -0.14 - 0.57*cos(phi(end:-1:1)) + 0.57*cos(phi(end:-1:1)).^2 + 1.14*cos(phi(end:-1:1)).^3; 
BP = BP/max(BP);

pos = [0.4 0.5 0.4 0.4];
figure('numbertitle','off','name','Expected beampattern',...
      'Units','normal','Position',pos);
plot(abs(BP));
ylabel('Gain');
xlabel('Incident angle');

%%
Nf = length(f);
x_map = zeros(N,Nf);
FI = zeros(Nphi,Nf);
figure(2)
for i=1:Nf
  Br = zeros(N,1);  
  Rc = (f(i)*N*dH/c);
  boudary  = find(abs(n)<= Rc  );
  phiS = acos(n(boudary)/Rc);
  Br(boudary)= -0.14 - 0.57*cos(phiS) + 0.57*cos(phiS).^2 + 1.14*cos(phiS).^3;
  Br(boudary) = abs(Br(boudary));
  idxBr = boudary(end);
  Br(idxBr+1:end)=1;
  idxBr = boudary(1);
  Br(1:idxBr)=Br(idxBr);
  plot(abs(Br)) 
  temp = fftshift(Br(end:-1:1));
  h = fftshift(ifft(temp(end:-1:1))); 
  x_map(:,i) = h;
   
  %hold on
  pause(0.01)
  
  % data for beam plot
  beta = 2*pi*f(i)/c;                  % wave number
  D = exp(1j*beta*x_array(ones(1,Nphi),:).*cos(phi(ones(N,1),:))');
  FI(:,i) = (D*(h));     
end
%%
pos = [0.5 0.5 0.4 0.4];
figure('numbertitle','off','name','beam pattern of full array',...
      'Units','normal','Position',pos);
surf(180/pi*phi,f,abs(FI)');
axis tight
set(gca,'XTick',[0 45 90 135 180]);
view([25,50]);
xlabel('Azimuth');
ylabel('Hz');
zlabel('Magnitude');
grid on
box on
set(gcf,'color','w');

pos(1) = pos(1) + 0.2; 
figure('numbertitle','off','name','weight spectrum of sensor array',...
      'Units','normal','Position',pos);
  
imagesc(f,n,abs(x_map));
ylabel('sensor index');
xlabel('Hz');
colorbar

X = bsxfun(@minus,x_map,mean(x_map));
% Do the PCA
[coeff,score,latent] = pca(X);

pos(1) = pos(1) + 0.2; 
figure('numbertitle','off','name','compressed weight spectrum',...
      'Units','normal','Position',pos);
imagesc(abs(score(:,1:4)))
ylabel('sensor index');
xlabel('Dimentional reduction of the frequencies');
set(gcf,'color','w');
colorbar
for i=1:N
  e(i) = norm(score(i,1:4));
end
maxScore =max(abs(e));
idxScore = find(e>gamma*maxScore);
idxScoreCom = find(e<=gamma*maxScore);

% K-means clustering
opts = statset('Display','final');
[idxMic, Cx, sumd, Dx] = kmeans(abs(score(idxScoreCom,1:4)),numCluster,'Replicates',2000,'Options',opts);
sum(sumd)
x_sparse = zeros(N,1);

idxSpare = idxScore;
x_sparse(idxSpare) = x_array(idxSpare);

for iCluster= 1:numCluster
    idxC = find(idxMic==iCluster);
    DisArray = Dx(idxC,iCluster);
    [closetMic,idxMin] = mink(DisArray,numPrsnt);

    x_sparse(idxScoreCom(idxC(idxMin)),1) = x_array(idxScoreCom(idxC(idxMin)));
    idxSpare = [idxSpare, idxScoreCom(idxC(idxMin))];   
end
x_sparse = x_sparse(idxSpare)';


pos = [0.5 0.8 0.4 0.1];
myFig=figure('numbertitle','off','name',' array layout (cm)','Units','normal',...
   'Position',pos,'Menubar','none');
plot(100*x_sparse,zeros(length(x_sparse),1),'o','MarkerEdgeColor','k','MarkerFaceColor','r',...
 'MarkerSize',8);

set(gca,'YTick',[]);
set(gcf,'color','w');
grid on
title(sprintf('geometry of arrays in cm, M = %d', length(x_sparse)),'FontSize',12); 
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');

Distance_SA = zeros(length(x_sparse),length(x_sparse));
for i=1:length(x_sparse)
   Distance_SA(i,:) =  abs(x_sparse(i)-x_sparse);
end   
Distance ={Distance_SA};
%%
array_config ={x_sparse};
Nsparse = length(x_sparse);
h_map = zeros(Nsparse,Nf);
DP = zeros(Nphi,Nf);
idxPhi_M = 121:180;
idxPhi_S = (1:120) ;

BP_M = (BP(idxPhi_M))';
BP_S=(BP(idxPhi_S))';
WNG = zeros(length(f),1);
BPE = zeros(length(f),1);
DF = zeros(length(f),1);
C1= ones(length(f),1)*0.003*length(idxPhi_M);
C2= ones(length(f),1)*0.004*length(idxPhi_S);
for iConfig=1:1
    x_opt = array_config{iConfig};
    for iF = 1:length(f)
        beta = 2*pi*f(iF)/c;             % wave number
        D = exp(-1j*beta*x_opt(ones(1,Nphi),:).*cos(phi(ones(Nsparse,1),:))');
        D_M = D(idxPhi_M,:);             % main lope
        D_S = D(idxPhi_S,:);             % side lope
        d = (D(180,:))  ;                % looking direction

        cvx_begin 
              variable xh(Nsparse) complex 
              minimize( norm(xh) )
              subject to 
                  (d*conj(xh)) == 1;                         % looking direction constraint
                   norm(BP_M - D_M*conj(xh)) <= C1(iF);      % main lope constraint
                   norm(BP_S - D_S*conj(xh)) <= C2(iF);      % side lope constraint
       cvx_end;
       h=xh;
       h_map(:,iF) = h;
       DP(:,iF)=D*h;
       WNG(iF) = xh'*xh;
       Shi = (sin(beta*Distance{iConfig})./(beta*Distance{iConfig}));
       Shi(logical(eye(size(Shi)))) = 1;
       DF(iF) = xh'*Shi*xh;
       if norm(h) < inf
           BPE(iF) = (sum(abs((BP_M) - (D_M*conj(h)))) + ...
           sum(abs((BP_S) - (D_S*conj(h)))))/length(BP);
       else
           BPE(iF) = inf;
       end
    end
    
    S.(sprintf('DP%d', iConfig)) = DP;
    S.(sprintf('WNG%d', iConfig))= WNG;
    S.(sprintf('DF%d', iConfig))= DF;
    S.(sprintf('BPE%d', iConfig))= BPE;
end
%%
pos = [0.1 0.1 0.4 0.4];
set(gcf,'defaultAxesFontSize',12)
 myFig =    figure('numbertitle','off','name','Beam pattern','Units','normal',...
       'Position',pos);
for iConfig=1:1
    p = linspace(0, pi, 180)';
    hs = surf(f,p*180/pi,-abs(20*log10(abs(S.(sprintf('DP%d', iConfig))) +eps)));
    ylabel('Azimuth');
    xlabel('Hz');
    zlabel('dB');
    xlim([fl fu]);
    set(hs, 'LineStyle','none');
    title(Plot_Title(iConfig));
    set(gcf,'color','w');
    set(gca,'FontSize', 12);
    grid on
    box on
end
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal')

% Plot White Noise Gain
pos = [0.1 0.1 0.4 0.4];
myFig = figure('numbertitle','off','name','White noise gain','Units','normal',...
       'Position',pos);
hold on
for iConfig=1:1
    plot(f,10*log10(1./S.(sprintf('WNG%d', iConfig))),strcat('-',Plot_Color{iConfig},Marker{iConfig}),'MarkerEdgeColor',Plot_Color{iConfig});
    xlabel('Hz');
    ylabel('WNG (dB)');
    legend('sparse array', 'small size uniform array', 'big size uniform array','incoherent design');
    set(gca,'FontSize', 12);
    set(gcf,'color','w');
    axis tight
    grid on
    box on
end
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal')

% Plot Directivity factor
pos(1) = pos(1) +0.1;
myFig = figure('numbertitle','off','name','Directivity factor','Units','normal',...
       'Position',pos);
hold on
for iConfig=1:1
    plot(f,10*log10(1./S.(sprintf('DF%d', iConfig))),strcat('-',Plot_Color{iConfig},Marker{iConfig}),'MarkerEdgeColor',Plot_Color{iConfig});
    xlabel('Hz');
    ylabel('DF (dB)');
    legend('sparse array', 'small size uniform array', 'big size uniform array','incoherent design');
    axis tight
    ylim([0 12.5])
    set(gcf,'color','w');
    set(gca,'FontSize', 12);
    grid on
    box on
end
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');

% Plot beampattern error everage
pos(1) = pos(1) +0.1;
figure('numbertitle','off','name','Beamp pattern error everage','Units','normal',...
       'Position',pos);
hold on
for iConfig=1:1
    plot(f,S.(sprintf('BPE%d', iConfig)),strcat('-',Plot_Color{iConfig},Marker{iConfig}),'MarkerEdgeColor',Plot_Color{iConfig});
    xlabel('Hz');
    ylabel('BPE');
    legend('sparse array', 'small size uniform array', 'big size uniform array','incoherent design');
    axis tight
    ylim([0 0.25])
    set(gca,'FontSize', 12);
    set(gcf,'color','w');
    grid on
    box on
end
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');

p = linspace(0, pi, 180);
k_p = round((1000-fl)/Fsh*Nfh)+1;
myFig = figure('numbertitle','off','name','beam pattern at 1 kHz','Units','normal',...
   'Position',pos);
for iConfig=1:1
    dBmin = -50;
    R_ref = abs(BP(end:-1:1))';
    RdB = max(dBmin,20*log10(R_ref));
    polarplot(p,RdB','k','LineWidth',1.5); 

    hold on

    R = abs(S.(sprintf('DP%d', iConfig))(:,k_p));
    RdB = max(dBmin,20*log10(R)) ;
    polarplot(p,RdB','--r','LineWidth',1.5); 

    axis tight;
    thetalim([0 180])
    title(Plot_Title(iConfig));
    set(gcf,'color','w');
    set(gca,'FontSize', 12);
end
legend('expected beam pattern','real beam pattern');
set(gcf,'color','w');
set(findall(myFig, 'Type', 'Text'),'FontWeight', 'Normal');
grid on
box on