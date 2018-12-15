function [ A ] = normalizeMatrix ( A, m ,s )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    for i=1:size(A,2)
        x = A(:,i);
        nx = (x(:) - m(i)) / s(i);
        A(:,i) = nx;
    end
end

