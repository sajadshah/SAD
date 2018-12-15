function [ x ] = isNotForTest( name, testArrayNames )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    x = 1;
    for i=1:length(testArrayNames)
        name1 = name;
        name2 = [testArrayNames(i).name,'.fea'];
        if( strcmp(name1, name2) )
            x=0;
            break;
        end
    end
end

