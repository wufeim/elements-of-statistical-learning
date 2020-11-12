Chapter 3
=====================================

**Exercise 3.11.** Show that the solution to the multivariate linear regression problem

.. math::

  \text{RSS}(\mathbf{B}; \mathbf{\Sigma}) = \sum_{i=1}^N (y_i - f(x_i))^\top \mathbf{\Sigma}^{-1}(y_i - f(x_i))

is given by

.. math::

  \hat{\mathbf{B}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X^\top Y}

What happens if the covariance matrices :math:`\mathbf{\Sigma}_i` are different for each observation?
