function [X Y] = circle(center, radius, NOP) % H=circle(center,radius,NOP,style)
%---------------------------------------------------------------------------------------------
% [X Y] = circle(center, radius, NOP)
% This routine draws a circle with center defined as
% a vector CENTER, radius as a scaler RADIS. NOP is 
% the number of points on the circle.
%
%   Usage Examples,
%
%   [X Y] = circle([1, 3], 3, 1000);
%   [X Y] = circle([2, 4], 2, 1000);
%   plot(X, Y);
%   axis equal;
%
%   Zhenhai Wang <zhenhai@ieee.org>
%   Version 1.00
%   December, 2002
%   
%   Modified by Alan Chu
%---------------------------------------------------------------------------------------------

if (nargin < 3),
 error('Please see help for INPUT DATA.');
elseif (nargin == 3)
    style = 'b-';
end;
THETA = linspace(0, 2*pi, NOP);
RHO = ones(1, NOP)*radius;
[X Y] = pol2cart(THETA, RHO);
X = X + center(1);
Y = Y + center(2);
plot(X, Y, style);
axis square;