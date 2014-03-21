gg = load('phog_features.mat');

arr = gg.arr;

imgs_test = dir('test');
 true_cor = 0;
 false_wr = 0;
%  [numimages,temp] = size(imgs_test);
 test_gist = zeros(1,2100);
 
 ntc = zeros(11,1);
    ntc(1,1) = 1;
    ntc(2,1) = 2;
    ntc(3,1) = 10;
    ntc(4,1) = 17;
    ntc(5,1) = 24;
    ntc(6,1) = 33;
    ntc(7,1) = 56;
    ntc(8,1) = 224;
    ntc(9,1) = 229;
    ntc(10,1) = 235;
    ntc(11,1) = 252;
 
 numimages = length(imgs_test);
 
for i = 3:numimages
%     img = imread(strcat('test/',imgs_test(i,:)));
    img = imread(['test/' imgs_test(i).name]);
    img_size = size(img);
    [temp,channel] = size(img_size);
    if channel == 3
        img = rgb2gray(img);
    end
     img = imresize(img, [400 400]);
     f =  anna_phog(img, 100, 180, 20, [1 size(img,1), 1, size(img,2)]','pie' );
     test_gist(1,:) = f(:);
    
    %figure,imshow(img);
%     inclass = imgs_test(i,1:3);
    %test_gist = LMgist(img,'',param);
   
    val = zeros(11,1);
    
    for j = 1:11
        gp = svmclassify(arr(j),test_gist);
        
        if gp == 1
            val(j,1) = 1;
        end
    end
    
    dt = find(val == 1);
    cls = ntc(dt);
    
    gt = str2num(imgs_test(i).name(1:3));
    
    if find(cls == gt)
        true_cor = true_cor + 1;
    else
        false_wr = false_wr + 1;
    end
    
end
true_cor/(true_cor + false_wr)
