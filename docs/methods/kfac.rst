KFAC
=====

KFAC is a simple :doc:`influence function <influence-functions>` with moderate accuracy and a fairly low computational cost.

KFAC's :math:`H` is a damped Kronecker-factored approximation to the Gauss-Newton Hessian. The :math:`H` is fitted on the training set with the fully trained model, so KFAC does not require access to training checkpoints or any other training artifacts.
