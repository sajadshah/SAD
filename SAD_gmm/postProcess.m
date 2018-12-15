function [ ppLabels ] = postProcess( testLabels )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    minOccurence = 5;
    besideOccurence = minOccurence*2;
    ppLabels = zeros(size(testLabels,1),1);
    before = 0;
    index =  1;
    c=0;
    while index < size(testLabels,1)
        endOfCurrent = nextChange(testLabels,index);        
        current = endOfCurrent - index;
        endOfAfter = nextChange(testLabels,endOfCurrent);
        after = endOfAfter - endOfCurrent;
        if( current < minOccurence && before > besideOccurence && after > besideOccurence)
            %here label ofsome frames which are less than minOccurence must change
            c=c+current;
            trueLabel = -testLabels(index); %opposite of previous label
            for i=index:endOfCurrent-1
                ppLabels(i) = trueLabel;
            end
        else
            for i=index:endOfCurrent-1
                ppLabels(i) = testLabels(i);
            end
        end
        before = current;
        index = endOfCurrent;
    end
    disp(c);
end

function [ nc ] = nextChange(list, index)
    nc = index+1;
    while (nc<size(list,1) && list(index)==list(nc))
        nc=nc+1;
    end
end
