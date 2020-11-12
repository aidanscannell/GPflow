import numpy as np
import gpflow as gpf


def test_separate_independent_conditional():
    """
    In response to bug #1523, fix separate_independent_conditional
    to prevent it from failing when q_sqrt=None.
    """
    def func(X):
        y1 = np.sin(X.flatten())
        y2 = np.cos(X.flatten())
        return np.stack([y1, y2]).T

    num_inducing = 30
    num_data = 100
    num_test = 200
    input_dim = 1

    X = np.linspace(0, 3, num_data).reshape(num_data, input_dim)
    Y = func(X)
    output_dim = Y.shape[1]

    # initialise SVGP with separate independent MOK and shared independent inducing variables
    idx = np.random.choice(range(num_data), size=num_inducing, replace=False)
    inducing_inputs = X[idx, ...].reshape(-1, input_dim)
    inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(inducing_inputs))
    kern_list = [gpf.kernels.RBF() for _ in range(output_dim)]
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    model = gpf.models.SVGP(kernel,
                            gpf.likelihoods.Gaussian(),
                            inducing_variable=inducing_variable,
                            num_latent_gps=output_dim)
    gpf.utilities.print_summary(model)

    # create test inputs
    Xnew = np.linspace(-1, 4, num_test).reshape(num_test, input_dim)

    # evaluate sparse gp conditional with q_sqrt=None
    mu, var = gpf.conditionals.conditional(Xnew,
                                           model.inducing_variable,
                                           model.kernel,
                                           model.q_mu,
                                           full_cov=False,
                                           full_output_cov=False,
                                           q_sqrt=None,
                                           white=model.whiten)


test_separate_independent_conditional()
