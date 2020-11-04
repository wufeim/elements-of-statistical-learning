3.1 Introduction
=====================================

A linear regression model assumes that the regression function $E(Y \mid X)$ is linear in the inputs $X_1, \dots, X_p$. They are simple and often provide an adequate and interpretable description of how the inputs affect the output.

3.2 Linear Regression Models and Least Squares
=====================================

We have an input vector $X^\top = (X_1, \dots, X_p)$, we want to predict a real-valued output $Y$. The linear regression model has the form

.. math::

  f(X) = \beta_0 + \sum_{j=1}^p X_j\beta_j

Here $X_j$ can be quantitative inputs, transformations of quantitative inputs, or numerical encoding of qualitative inputs.
