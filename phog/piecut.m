function [ mask ] = piecut( radius, cuts )

rings = [];

upperRadius = radius;
lowerRadius = 0;

mask = zeros(2*radius,2*radius);
[pX pY] = meshgrid(-radius:radius, -radius:radius);
pA = (atan(pX./pY));
pA = pA + [repmat(pi/2, [radius 2*radius+1]); repmat(3*pi/2, [radius+1 2*radius+1])];

pR = sqrt(pX(:).^2+pY(:).^2);

cutIndex = 1; 
cutArea = radius^2/sum(cuts);

for cut = 1:length(cuts)
                
        if cut<length(cuts)
                lowerRadius = sqrt(upperRadius^2 - cuts(cut) * cutArea);
        else
                lowerRadius = 0;
        end

        segAngle = deg2rad(360/cuts(cut));

        for seg=0:cuts(cut)-1

                angles = [seg*segAngle (seg+1)*segAngle];

                for i=1:length(pX)*length(pY)   

                        if pA(i)>angles(1) &&  pA(i)<=angles(2) && pR(i)<upperRadius && pR(i)>lowerRadius
                                mask(pY(i)+radius+1, pX(i)+radius+1) = cutIndex;
                        end

                end
                
                cutIndex = cutIndex + 1;

        end
        
        rings = [rings lowerRadius];
        upperRadius = lowerRadius;
        lowerRadius = 0;

end

end 
