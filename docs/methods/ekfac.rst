EKFAC
=====

EKFAC is the workhorse of :doc:`influence functions <influence-functions>`, providing fairly good accuracy and a fairly low computational cost.

EKFAC's :math:`H` is a damped Kronecker-factored approximation to the Gauss-Newton Hessian. The :math:`H` is fitted on the training set with the fully trained model, so EKFAC does not require access to training checkpoints or any other training artifacts. Compared to KFAC, EKFAC's :math:`H` applies an additional correction step, where the Kronecker product's eigenvalues are set to the second moments of the gradients in the Kronecker eigenbasis.
