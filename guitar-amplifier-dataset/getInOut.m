function [train,test] = getInOut(modelName,lStyle,lGain,trainTestRatio)
%%% return a matrix with the selected input sounds,output sounds, gain
%%% Parameters in columns 1,2,3 respectively

%Parameters:
% modelName : list of char with the selected amplifier. ex:
% 'EnglDisto.wav'for the Engl retrotube 50
% or 'IbanezTSA15.wav' for the Tube Screamer 15h.
% lStyle : a vector containing the wished samples. ex: [4,2] will
% return the concatenation of the sample 4 (bluesMILASI) and then the
% sample 2 (chordMajeur).
%lGain : a vector containing a list of Gains (parameter of the
%amplifier) from 1 to 10 (10 is the highest distorsion level of the
%amplifier)
%trainTestRatio: set the ratio of data that have to gone in the training-set and in the test-set

% example of use : mat = getInOut('EnglDisto.wav',[5,3],[1,10],0.8); return two matrix of
% three columns where the first one is the input sound, the second one the
% output sound (from the amplifier) and the third one is the gain
% parameters. The matrix will contain the concatenation of the sample 5 and
% 3 for the gain parameter 1 then for the parameters 10. Matrix "training"
% will contains 80% of the dataset and matrix "Test" will contains 20% of
% the dataset.

% Author: Thomas Schmitz 2018, contact T.Schmitz@ulg.ac.be

if (nargin < 4)
    trainTestRatio = 0.8;
end

inAmp = audioread('GuitarMixIn.wav'); %
outAmp = audioread(modelName); %
disp('Processing of the synchronization of the files, please wait')
[inAmp, outAmp] = synchronize(inAmp,outAmp); % synchronization by correlation method, comment if it is not required
trans=2*44100; % duration of silence between samples
x1 = audioread('chromaticNotesIn.wav'); % Sample 1
x2 = audioread('chordMajeurMinorIn.wav'); % Sample 2
x3 = audioread('bambaChordIn.wav'); % Sample 3
x4 = audioread('bluesMiLaSiChordandNotes.wav'); % Sample 4
x5 = audioread('scaleAmNotesIn.wav'); % Sample 5
lengthAudio=[length(x1),length(x2), length(x3),length(x4),length(x5)];
%lengthAudio = [4581376 1684480 680389 910336 551424];
inOutAmp = [inAmp,outAmp];
train=[];
test=[];
rng(66) %random same prediction

for k= 1: length(lGain)
    for i = 1: length(lStyle)
        neededSample = lStyle(i);
        sampleBefore = [neededSample-1:-1:1];
        count=0;
        for j = 1:length(sampleBefore)
            count = count + lengthAudio(j); 
        end
        start = trans*(neededSample-1)+ count +1 + (lGain(k)-1)*(5*trans+sum(lengthAudio));
        stop = start + lengthAudio(neededSample);
        
        durationTest = floor((stop-start)*(1-trainTestRatio));
        possibleRand = floor((stop-start)/durationTest)  ; 
        r = randi([1 possibleRand]);
        
    train = [train;inOutAmp(start:start+(r-1)*durationTest-1,:),lGain(k).*ones((r-1)*durationTest,1)];
    train = [train;inOutAmp(start+r*durationTest+1:stop,:),lGain(k).*ones(stop-start-r*durationTest,1)];
    test = [test;inOutAmp(start+(r-1)*durationTest:start+r*durationTest,:),lGain(k).*ones(durationTest+1,1)];
    
    end
end

% Save in .mat
save(['training' modelName(1:end-4) '_Gain5.mat'],'train')
save(['test' modelName(1:end-4) '_Gain5.mat'],'test')
disp('Done, enjoy !')