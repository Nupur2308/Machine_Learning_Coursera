function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

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

z = X*theta;
h = sigmoid(z);
jj = (y'*log(h)) + ((1-y)'*log(1-h));
theta_sub = theta(2:size(theta,1));
reg_term = lambda*(theta_sub'*theta_sub)/(2*m);
J = -jj/m + reg_term;

delta = X'*(h-y);
grad = delta/m;
grad(2:size(theta,1)) += lambda/m*theta(2:size(theta,1));

% =============================================================

end
