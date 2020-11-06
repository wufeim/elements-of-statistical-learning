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

3.2.1 Example: Prostate Cancer
-------------------------------------

3.2.2 The Gauss-Markov Theorem
-------------------------------------

We focus on estimation of any linear combination of the parameters :math:`\theta = a^\top\beta`. The least squares estimate of :math:`a^\top\beta` is

.. math::

  \hat{\theta} = a^\top\hat{\beta} = a^\top (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}

Considering :math:`\mathbf{X}` to be fixed, this is a linear function :math:`\mathbf{c}_0^\top\mathbf{y}` of the response vector :math:`\mathbf{y}`. If we assume that the linear model is correct, :math:`a^\top\hat{\beta}` is unbiased since

.. math::

  \text{E}(a^\top\hat{\beta}) & = \text{E}(a^\top (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}) \nonumber \\
	& = a^\top (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \text{E}(\mathbf{y}) \nonumber \\
	& = a^\top \beta

*The Gauss-Markov theorem* states that if we have any other linear estimator :math:`\tilde{\theta} = \mathbf{c}^\top\mathbf{y}` that is unbiased for :math:`a^\top\beta`, that is :math:`\text{E}(\mathbf{c}^\top\mathbf{y}) = a^\top\beta`, then

.. math::

  \text{Var}(a^\top\hat{\beta}) \leq \text{Var}(\mathbf{c}^\top\mathbf{y})

For simplicity we have stated the result in terms of estimation of a single parameter :math:`a^\top\beta`. The proof is left as Exercise 3.3 and is given in my solution manual.

Consider the mean squared error of an estimator :math:`\tilde{\theta}` in estimating :math:`\theta`:

.. math::

  \text{MSE}(\tilde{\theta}) & = \text{E}(\tilde{\theta} - \theta)^2 \nonumber \\
	& = \text{E}(\tilde{\theta} - \text{E}(\tilde{\theta}) + \text{E}(\tilde{\theta}) - \theta)^2 \nonumber \\
	& = \text{E}\left[(\tilde{\theta} - \text{E}(\tilde{\theta}))^2 + 2 (\tilde{\theta} - \text{E}(\tilde{\theta}))(\text{E}(\tilde{\theta}) - \theta) + (\text{E}(\tilde{\theta}) - \theta)^2 \right] \nonumber \\
	& = \text{E}(\tilde{\theta} - \text{E}(\tilde{\theta}))^2 + 2 (\text{E}(\tilde{\theta}) - \theta) \text{E}(\tilde{\theta} - \text{E}(\tilde{\theta})) + [\text{E}(\tilde{\theta}) - \theta]^2 \nonumber \\
	& = \text{Var}(\tilde{\theta}) + [\text{E}(\tilde{\theta}) - \theta]^2

The first term is the variance, while the second term is the squared bias.

The Gauss-Markov theorem implies that the least squares estimator has the smallest mean squared error of all linear estimators with no bias. However, there may well exist a biased estimators with smaller mean squared error. Such an estimator would trade a little bias for a larger reduction in variance. We discuss many examples, including variable subset selection and ridge regression, later in this chapter. From a more pragmatic point of view, most models are distortions of the truth, and hence are biased; picking the right model amounts to creating the right balance between bias and variance.

Consider the prediction of the new response at input :math:`x_0`,

.. math::

  Y_0 = f(x_0) + \varepsilon_0

Then the expected prediction error of an estimate :math:`\tilde{f}(x_0) = x_0^\top \tilde{\beta}` is

.. math::

  \text{E}(Y_0 - \tilde{f}(x_0))^2 & = \text{E}(f(x_0) + \varepsilon_0 - x_0^\top \tilde{\beta})^2 \nonumber \\
	& = \text{E}(\varepsilon_0^2) + 2\text{E}(\varepsilon_0(f(x_0) - x_0^\top\tilde{\beta})) + \text{E}(x_0^\top\tilde{\beta} - f(x_0))^2 \nonumber \\
	& = \sigma^2 + \text{E}(x_0^\top\tilde{\beta} - f(x_0))^2 \nonumber \\
	& = \sigma^2 + \text{MSE}(\tilde{f}(x_0))

Therefore, expected prediction error and mean squared error differ only by the constant :math:`\sigma^2`, representing the variance of the new observation :math:`y_0`.
