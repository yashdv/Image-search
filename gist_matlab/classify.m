function classVal = classify(fname)

mod = load('trained_model.mat');
s = mod.s;

img = imread(fname);

clear param
param.orientationsPerScale = [8 8 8 8]; 
param.numberBlocks = 4;
param.fc_prefilt = 4;

[gist, param] = LMgist(img, '', param); 
val = zeros(11,1);

for i=1:11
    val(i,1) = svmclassify(s(i),gist);
end
classVal = find(val == 1);
end