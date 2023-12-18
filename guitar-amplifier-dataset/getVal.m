function [val] = getVal(modelName,listSamples,listGain)
%%% return a matrix with the selected input sounds,output sounds, gain Parameters

%Parameters:
% modelName : list of char with the selected amplifier. ex:
% 'EnglOut.wav'for the Engl retrotube 50
% or 'TSA15Out.wav' for the Tube Screamer 15h.
% listSamples : a vector containing the wished samples. ex: [4,2] will
% return the concatenation of the sample 4 (bluesMILASI) and then the
% sample 2 (chordMajeur).
%listGain : a vector containing a list of Gains (parameter of the
%amplifier) from 1 to 10 (10 is the highest distorsion level of the
%amplifier)

% example of use : mat = getInOut('EnglOut.wav',[5,3],[1,10]); return a matrix of
% three column where the first one is the input sound, the second one the
% output sound (from the amplifier) and the third one is the gain
% parameters. The matrix will contain the concatenation of the sample 5 and
% 3 for the gain parameter 1 then for the parameters 10
% Thomas Schmitz 2018


trainTestRatio = 1;

inAmp = audioread('GuitarMixIn.wav'); %
outAmp = audioread(modelName); %
[inAmp, outAmp] = synchronize(inAmp,outAmp);
trans=2*44100; % duration of silence between samples
x1 = audioread('chromaticNotesIn.wav'); %
x2 = audioread('chordMajeurMinorIn.wav'); % (Do Re Mi Fa Sol La Si major then minor)
x3 = audioread('bambaChordIn.wav'); %
x4 = audioread('bluesMiLaSiChordandNotes.wav'); %
x5 = audioread('scaleAmNotesIn.wav'); %
lengthAudio=[length(x1),length(x2), length(x3),length(x4),length(x5)];
inOutAmp = [inAmp,outAmp];
train=[];

for k= 1: length(listGain)
    for i = 1: length(listSamples)
        neededSample = listSamples(i);
        sampleBefore = [neededSample-1:-1:1];
        count=0;
        for j = 1:length(sampleBefore)
            count = count + lengthAudio(j); 
        end
        start = trans*(neededSample-1)+ count +1 + (listGain(k)-1)*(5*trans+sum(lengthAudio));
        stop = start + lengthAudio(neededSample);
        

    train = [train;inOutAmp(start:stop,:),listGain(k).*ones(stop-start+1,1)];
    
    
    end
end
val =train;
% Save in .mat
save(['val' modelName(1:end-4) '_Gain5.mat'],'val')
