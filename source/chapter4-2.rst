4.2 Linear Regression of an Indicator Matrix
=====================================

Suppose :math:`\mathcal{G}` has :math:`K` classes, each response category is associated with an indicator :math:`Y_k` with :math:`Y_k = 1` if :math:`G = k` else :math:`0`. These are collected together in a vector :math:`Y = (Y_1, \dots, Y_k)`, and the :math:`N` training instances of these form an :math:`N \times K` **indicator response matrix** :math:`\mathbf{Y}`. We fit a regression model to each of the columns of :math:`\mathbf{Y}` simultaneously, and the fit is given by

.. math::

   \hat{\mathbf{Y}} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top\mathbf{Y}

From which we obtain the :math:`(p+1) \times K` coefficient matrix :math:`\hat{\mathbf{B}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top\mathbf{Y}`. A new observation with input :math:`x` is classified by first computing the fitted output :math:`\hat{f}(x)^\top = (1, x^\top)\hat{\mathbf{B}}` and then predicting the category by identifying the largest component: :math:`\hat{G}(x) = \text{argmax}_{k \in \mathcal{G}}\hat{f}_k(x)`.
