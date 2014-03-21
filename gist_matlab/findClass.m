function classVal = findClass(gist,s)

val = zeros(11,1);

for i=1:11
    val(i,1) = svmclassify(s(i),gist);
end
classVal = find(val == 1);
end