list = dir('testing_data');

len = length(list);

Nimages = 330;

clear param
param.imageSize = [256 256]; % set a normalized image size
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Pre-allocate gist:
Nfeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;
gist = zeros([Nimages Nfeatures]);
labels = zeros(Nimages,1);

% Load first image and compute gist:
folder_name = ['testing_data/' list(3).name];
list1 = dir(folder_name);
fname = [folder_name '/' list1(3).name];
img = imread(fname);
[gist(1, :), param] = LMgist(img, '', param); % first call
labels(1,1) = str2num(list(3).name);
% Loop:
k = 2;
for i = 3:len
    folder_name = ['testing_data/' list(i).name];
    list1 = dir(folder_name);
    ll = length(list1);
    for j = 3:ll
        if i==3 && j==3
            i
            j
            continue;
        end
        fname = [folder_name '/' list1(j).name];
        img = imread(fname);
        gist(k, :) = LMgist(img, '', param); % the next calls will be faster
        labels(k,1) = str2num(list(i).name);
        k = k+1;
    end
end
save('new_features.mat','gist','labels');