function [inSync outSync] =  synchronize(in,out)


crossLength=10*44100;


%y = audioread([name '.wav']);
[c,lags] = xcorr(in(1:crossLength),out(1:crossLength));
[~,I]=max(c);

[c,lags] = xcorr(in(1:crossLength),-out(1:crossLength));
[~,J]=max(c);


if I>J % case where the signal is not inverted
    s = lags(I);
    inSync =  in(1:end+s);
    outSync = out(-s+1:end);
else % case where the signal is inverted
    s = lags(J);
    inSync =  in(1:end+s);
    outSync = -out(-s+1:end);
end




    

%save('trainingAll','mat')
%save('validationAll','validation')

%%
%figure,

%hold all
    
%    p(1) = plot(xCorrected(44100:44100+1000),'-');
%    p(2) = plot(yCorrected(44100:44100+1000),'x');  