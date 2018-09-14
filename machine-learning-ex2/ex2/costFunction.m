function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
X_theta = X * theta;
h_X_theta = sigmoid(X_theta);

logX = [log(h_X_theta) log(1-h_X_theta)];
Y = [-y -(1-y)];
X_logX = Y * logX';

J = sum(diag(X_logX))/m;
%grad = X' * (h_X_theta) ./ m;
grad = (1/m)*(X')*(h_X_theta - y);


% =============================================================

end
