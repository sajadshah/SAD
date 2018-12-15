function [ nx ] = normalizeColumnVector( x )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    m = mean(x,1);
    s = std(x,1,1);
    nx = (x(:) - m) / s;
end

