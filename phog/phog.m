function [ features ] = phog( images, bins, L, mode )

features = [];

for group=1:length(images)
        fprintf('Analyzing group %d...\n', group);
        for image=1:length(images{group})
                img = imresize(images{group}{image}, [400 400]);
                f = anna_phog(img, bins, 180, L, [1 size(img,1), 1, size(img,2)]', mode);
                features(image, :, group) = f;
        end
end

end