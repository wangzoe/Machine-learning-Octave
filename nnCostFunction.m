function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables

m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%part one
a1 = [ones(m,1) X];
z2 = a1 * transpose(Theta1);
a2 = [ones(size(z2), 1) sigmoid(z2)];
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);

delta1 = 0;
delta2 = 0;

reg_Theta1 = Theta1(:,2:end);
reg_Theta2 = Theta2(:,2:end);
reg = (sum(sum(reg_Theta1.^2))+sum(sum(reg_Theta2.^2)))*lambda/2/m;


for c = 1:num_labels
  y_class = y==c;
  h = a3(:,c);% take out the vector of y==c in a3, a3 is 5000*10, y is 5000*1
  J = J + ...
  1/m * sum((-y_class.*log(h) - (1-y_class).*log(1-h)));
end

J = J + reg;

% part 2
for i = 1:m
  a_1 = transpose(X(i, :));
  a_1 = [1; a_1];
  z_2 = Theta1 * a_1;
  a_2 = [1; sigmoid(z_2)];
  z_3 = Theta2 * a_2;
  a_3 = sigmoid(z_3);
  
  [p, h] = max(a_3); % find out the predict number;
  y_class = zeros(num_labels,1);
  y_class(y(i)) = 1;
  
  sigma3 = a_3 - y_class;
  sigma2 = transpose(Theta2) * sigma3 .* sigmoidGradient([1;z_2]);
  
  sigma2 = sigma2(2:end);
  
  delta1 += sigma2 * transpose(a_1);
  delta2 += sigma3 * transpose(a_2);
end

D1 = delta1./m;
D2 = delta2./m;

% part 3
d_reg1 = lambda/m * reg_Theta1;
d_reg2 = lambda/m * reg_Theta2;

temp1 = D1(:,2:end) + d_reg1;
temp2 = D2(:,2:end) + d_reg2;

Theta1_grad = [D1(:,1) temp1];
Theta2_grad = [D2(:,1) temp2];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
