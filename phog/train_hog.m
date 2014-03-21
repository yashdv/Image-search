imgs = dir('train');
[nim , temp ] = size(imgs);
n = length(imgs);
phog = zeros([n-2 2100]); 
options = optimset('maxiter',30000);
for i = 3:n
    i
    img = imread(['train/' imgs(i).name]);
    img_size = size(img);
    [temp,channel] = size(img_size);
    if channel == 3
        img = rgb2gray(img);
    end
     img = imresize(img, [400 400]);
     f =  anna_phog(img, 100, 180, 20, [1 size(img,1), 1, size(img,2)]','pie' );
     phog(i-2,:) = f(:);
    
end
%[num_classes,temp] = size(imgs);
 
for j = 1:11
    grp = -1*ones(n-2,1);
    grp((j-1)*30+1:j*30,1) = 1;
    SVMStruct = svmtrain(phog,grp,'Method','QP','quadprog_opts',options);
    arr(j) = SVMStruct;
end

save('phog_features.mat','arr');

