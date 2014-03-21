function p = pie_phogDescriptor(bh,bv,L,bin)

p = [];

for b=1:bin
    ind = bh==b;
    p = [p;sum(bv(ind))];
end

for l=1:size(L,2),
   mask = piecut(size(bh,1)/2, L(1:l)); 
   
   for piece=1:sum(L(1:l))
       
       for b=1:bin
           seg = mask==piece;
           ind = (bh.*seg)==b;
           p = [p;sum(bv(ind))];
           
       end
       
   end
   
end


if sum(p)~=0
    p = p/sum(p);
end

end