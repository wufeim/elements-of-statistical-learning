3.5 Methods Using Derived Input Directions
=====================================

The methods in this section produce a small number of linear combinations :math:`Z_m`, :math:`m = 1, \dots, M` of a large number of correlated inputs :math:`X_j`, and the :math:`Z_m` are then used in place of the :math:`X_j` as inputs in the regression.

3.5.1 Principal Components Regression
-------------------------------------

In this approach the linear combinations :math:`Z_m` used are the principal components defined in Section 3.4.1. Principal components regression (PCR) depend on the scaling of the inputs, so typically we first standardize them. Since the :math:`\mathbf{z}_m` are orthogonal, the regression is just a sum of univariate regressions:

.. math::

  \hat{\mathbf{y}}_{(M)}^\text{pcr} = \bar{y}\mathbf{1} + \sum_{m=1}^M \hat{\theta}_m\mathbf{z}_m, \;\;\; \hat{\theta}_m = \frac{\langle \mathbf{z}_m, \mathbf{y} \rangle}{\langle \mathbf{z}_m, \mathbf{z}_m \rangle}

This solution can also be expressed in terms of the :math:`\mathbf{x}_j` (Exercise 3.13) where

.. math::

  \hat{\beta}^\text{pcr}(M) = \sum_{m=1}^M \hat{\theta}_mv_m

For :math:`M < p` we get a reduced regression. PCR is very similar to ridge regression: both operate via the principal components of the input matrix. Ridge regression shrinks the coefficients of the principal components, while PCR discards the :math:`p - M` components, as depicted in the figure below.

.. image:: images/fig3-8.png
  :width: 320pt

3.5.2 Partial Least Squares
-------------------------------------

Partial least squares (PLS) is not scale invariant, so we assume that each :math:`\mathbf{x}_j` is standardized to have mean 0 and variance 1.

| **Algorithm 3.3** *Partial Least Squares.*
|     1. Standardize each :math:`\mathbf{x}_j` to have mean zero and variance one. Set :math:`\hat{\mathbf{y}}^{(0)} = \bar{y}\mathbf{1}`, and :math:`\mathbf{x}_j^{(0)} = \mathbf{x}_j`, :math:`j = 1, \dots, p`.
|     2. For :math:`m = 1, \dots, p`
|         (a) :math:`\mathbf{z}_m = \sum_{j=1}^p \hat{\phi}_{mj}\mathbf{x}_j^{(m-1)}`, where :math:`\hat{\phi}_{mj} = \langle \mathbf{x}_j^{(m-1)}, \mathbf{y}\rangle`.
|         (b) :math:`\hat{\theta}_m = \langle \mathbf{z}_m, \mathbf{y} \rangle/\langle \mathbf{z}_m, \mathbf{z}_m\rangle`.
|         (c) :math:`\hat{\mathbf{y}}^{(m)} = \hat{\mathbf{y}}^{(m-1)} + \hat{\theta}_m\mathbf{z}_m`.
|         (d) :math:`\hat{\mathbf{y}}^{(m)} = \hat{\mathbf{y}}^{(m-1)} + \hat{\theta}_m\mathbf{z}_m`.
|         (e) Orthogonalize each :math:`\mathbf{x}_j^{(m-1)}` with respect to :math:`\mathbf{z}_m`: :math:`\mathbf{x}_j^{(m)} = \mathbf{x}_j^{(m-1)} - [\langle \mathbf{z}_m, \mathbf{x}_j^{(m-1)}\rangle / \langle \mathbf{z}_m, \mathbf{z}_m \rangle]\mathbf{z}_m`, :math:`j = 1, \dots, p`.
|     3. Output the sequence of fitted vectors :math:`\{\hat{\mathbf{y}}^{(m)}\}_1^p`. Since the :math:`\{\mathbf{z}_l\}_1^m` are linear in the original :math:`\mathbf{x}_j`, so is :math:`\hat{\mathbf{y}}^{(m)} = \mathbf{X}\hat{\beta}^\text{pls}(m)`. These linear coefficients can be recovered from the sequence of PLS transformations.

What optimization problem is PLS solving? It can be shown (Exercise 3.15) that PLS seeks directions that have high variance and have high correlation with the response, in contrast to PCR which keys only on high variance. In particular, the :math:`m` th principal component direction :math:`v_m` solves:

.. math::

  \max_\alpha \; & \text{Var}(\mathbf{X}\alpha) \\
	\text{subject to} \; & \lVert \alpha \rVert = 1, \; \alpha^\top \mathbf{S}v_l = 0, \; l = 1, \dots, m-1

where :math:`\mathbf{S}` is the sample covariance matrix of the :math:`\mathbf{x}_j`. The conditions :math:`\alpha^\top \mathbf{S}v_l = 0` ensures that :math:`\mathbf{z}_m = \mathbf{X}\alpha` is uncorrelated with all the previous linear combinations :math:`\mathbf{z}_l = \mathbf{X}v_l`. The :math:`m`th PLS direction :math:`\hat{\phi}_m` solves:

.. math::

  \max_\alpha \; & \text{Corr}^2 (\mathbf{y}, \mathbf{X}\alpha)\text{Var}(\mathbf{X}\alpha) \\
	\text{subject to} \; & \lVert \alpha \rVert = 1, \; \alpha^\top \mathbf{S}\hat{\phi}_l = 0, \; l = 1, \dots, m-1

Further analysis reveals that the variance aspect tends to dominate, and so PLS behaves much like ridge regression and PCR.

.. warning::

  Add some explanations here.

If the input matrix :math:`\mathbf{X}` is orthogonal, then PLS finds the least squares estimates after :math:`m = 1` steps. Subsequent steps have no effect since :math:`\hat{\phi}_{mj}` are zero for :math:`m > 1` (Exercise 3.14). It can also be shown that the sequence of PLS coefficients for :math:`m = 1, \dots, p` represents the conjugate gradient sequence for computing the least squares solutions (Exercise 3.18).

.. warning::

  Add solutions to Exercise 3.14 and Exercise 3.18.
