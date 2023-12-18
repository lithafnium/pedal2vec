% Code generator Tschmitz June 2018%
% Guitar prs on micro bridge
% Steinberg UR22 sound card 44100 32bit float


clear all
numberParameters = 10; % number of repeated input to estimate all the parameters
fs = 44100;
x1 = audioread('chromaticNotesIn.wav'); %
x2 = audioread('chordMajeurMinorIn.wav'); % (Do Re Mi Fa Sol La Si major then minor)
x3 = audioread('bambaChordIn.wav'); %
x4 = audioread('bluesMiLaSiChordandNotes.wav'); %
x5 = audioread('scaleAmNotesIn.wav'); %

trans = zeros(fs*2,1);
xIn = [x1;trans;x2;trans;x3;trans;x4;trans;x5;trans];

concatIn=[]
for i =1:numberParameters
    concatIn =[concatIn;xIn] ;
end;

audiowrite('guitarMixIn.wav',concatIn,fs,'bitsPerSample',32);

%%


