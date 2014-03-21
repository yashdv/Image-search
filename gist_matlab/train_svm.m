features = load('new_features.mat');

gist = features.gist;
labels = features.labels;

labelLen = length(labels);


uniLabels = unique(labels);
len = length(uniLabels);

for i=1:len
    group = zeros(labelLen,1);
    lbl = uniLabels(i);
    group(find(labels == lbl)) = 1;
    SVMStruct = svmtrain(gist,group,'Method','QP','quadprog_opts',options);
    s(i) = SVMStruct;
end

save('new_trained_model.mat','s');