function p = anna_PhogDescriptor(bh,bv,L,bin)
% anna_PHOGDESCRIPTOR Computes Pyramid Histogram of Oriented Gradient over a ROI.
%               
%IN:
%       bh - matrix of bin histogram values
%       bv - matrix of gradient values 
%   L - number of pyramid levels
%   bin - number of bins
%
%OUT:
%       p - pyramid histogram of oriented gradients (phog descriptor)

p = zeros(1, bin * sum((4*ones(L)).^0:L));
        
for l=0:L
    x = fix(size(bh,2)/(2^l));
    y = fix(size(bh,1)/(2^l));

        for xx=1:x:size(bh,2)
                for yy=1:y:size(bh,1)
                        bh_cella = bh(yy:yy+y-1,xx:xx+x-1);
            bv_cella = bv(yy:yy+y-1,xx:xx+x-1);
            
            for b=1:bin
                ind = bh_cella==b;
                                p(1,4^l*bin+b) = sum(bv_cella(ind));
            end 
                end        
        end
end

if sum(p)~=0
    p = p/sum(p);
end

end