3.2 Linear Regression Models and Least Squares
=====================================

We have an input vector :math:`X^\top = (X_1, \dots, X_p)`, we want to predict a real-valued output :math:`Y`. The linear regression model has the form

.. math::

  f(X) = \beta_0 + \sum_{j=1}^p X_j\beta_j

Here :math:`X_j` can be quantitative inputs, transformations of quantitative inputs, or numerical encoding of qualitative inputs.

Typically we have a set of training data :math:`(x_1, y_1), \dots, (x_N, y_N)` from which to estimate the parameters :math`\beta`. The most popular estimation method is *least squares*, in which we pick the coefficients :math:`\beta = (\beta_0, \dots, \beta_p)^\top` to minimize the residual sum of squares

.. math::

  \text{RSS}(\beta) = \sum_{i=1}^N (y_i - f(x_i))^2 = \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij}\beta_j)^2

From a statistical point of view, this criterion is reasonable if the training observations $(x_i, y_i)$ represent independent draws from their population. Even if the :math:`x_i`'s were not drawn randomly, the criterion is still valid if the :math:`y_i`'s are conditionally independent given the inputs :math:`x_i`.

Let :math:`\mathbf{X}` and :math:`\mathbf{y}` be the matrix representation of the training data. We can write the residual sum-of-squares as

.. math::

  RSS(\beta) = (\mathbf{y} - \mathbf{X}\beta)^\top (\mathbf{y} - \mathbf{X}\beta)

Differentiating w.r.t. :math:`\beta` we obtain

.. math::

  \frac{\partial \text{RSS}}{\partial \beta} & = -2\mathbf{X}^\top (\mathbf{y} - \mathbf{X}\beta) \\
  \frac{\partial^2 \text{RSS}}{\partial\beta\partial\beta^\top} & = 2\mathbf{X}^\top\mathbf{X}
