import time

import cupy as cp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin


class LSSVMRegression(BaseEstimator, RegressorMixin):
    """

    Attributes:
        - gamma     : the hyper-param (float)
        - kernel    : the kernel used (string)
        - kernel_   : the actual kernel function
        - x         : training data
        - y         : target of training data
        - coef_     : coeff of the support vectors
        - intercept_: intercept term

    """

    def __init__(
        self,
        gamma: float = 1.0,
        kernel: str = "rbf",
        c: float = 1.0,
        d: float = 2.0,
        sigma: float = 1.0,
        m: int = 101,
        x_init=None,
    ):

        # PARAMETER
        self.gamma = gamma
        self.c = c
        self.d = d
        self.sigma = sigma
        self.m = m
        self.x_init = x_init

        if kernel is None:
            self.kernel = "rbf"
        else:
            self.kernel = kernel

        params = dict()
        if kernel == "poly":
            params["c"] = c
            params["d"] = d
        elif kernel == "rbf":
            params["sigma"] = sigma

        self.kernel_ = LSSVMRegression.__set_kernel(self.kernel, **params)
        self.B = np.zeros(m + 1, dtype=float)
        # SAVING INVERSION MATRIX
        self.omega = self.kernel_(self.x_init, self.x_init)
        self.inv = LSSVMRegression.inverse(self, self.x_init, self.m)

        # MODEL PARAM
        self.x = None
        self.y = None
        self.coef_ = None
        self.intercept_ = None

    def get_params(self, deep=True):
        return {
            "c": self.c,
            "d": self.d,
            "gamma": self.gamma,
            "kernel": self.kernel,
            "sigma": self.sigma,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        params = dict()
        if self.kernel == "poly":
            params["c"] = self.c
            params["d"] = self.d
        elif self.kernel == "rbf":
            params["sigma"] = self.sigma

        self.kernel_ = LSSVMRegression.__set_kernel(self.kernel, **params)

        return self

    def set_attributes(self, **parameters):
        for param, value in parameters.items():
            if param == "intercept_":
                self.intercept_ = value
            elif param == "coef_":
                self.coef_ = value
            elif param == "support_":
                self.x = value

    @staticmethod
    def __set_kernel(name: str, **params):
        def linear(xi, xj):
            return np.dot(xi, xj.T)

        def poly(xi, xj, c=params.get("c", 1.0), d=params.get("d", 2)):
            return ((cp.dot(xi, xj.T)) / c + 1) ** d

        def rbf(xi, xj, sigma=params.get("sigma", 1.0)):
            if xi.ndim == 2 and xi.ndim == xj.ndim:
                return np.exp(-(cdist(xi, xj, metric="sqeuclidean")) / (2 * (sigma**2)))
            elif (xi.ndim < 2) and (xj.ndim < 3):
                ax = len(xj.shape) - 1
                return np.exp(
                    -(np.dot(xi, xi) + (xj**2).sum(axis=ax) - 2 * np.dot(xi, xj.T))
                    / (2 * (sigma**2))
                )
            else:
                message = "The rbf kernel is not suited for arrays with rank > 2"
                raise Exception(message)

        kernels = {"linear": linear, "poly": poly, "rbf": rbf}
        if kernels.get(name) is not None:
            return kernels[name]
        else:
            message = "Kernels" + name + "is not implemented. Please choose from : "
            message += str(list(kernels.keys())).strip("[]")
            raise KeyError(message)

    def inverse(self, x, m):
        Omega = self.kernel_(x, x)
        Ones = np.array([[1]] * m, dtype=float)
        A = np.block([[0, Ones.T], [Ones, Omega + (self.gamma**-1) * np.identity(m)]])

        st = time.monotonic()
        A_dag = cp.linalg.inv(cp.asarray(A))
        et = time.monotonic()
        s = et - st
        print(f"TIME INVERSE : {s:.5f} s\n")

        return A_dag

    def __OptimizeParams(self):
        self.B[1:] = self.y

        solution = cp.dot(
            cp.asarray(self.inv.astype(float)).get(),
            cp.asarray(self.B.astype(float)).get(),
        )

        self.intercept_ = solution[0]
        self.coef_ = solution[1:]

    def fit(self, X: cp.ndarray, y: cp.ndarray):
        self.x = X
        self.y = y
        self.__OptimizeParams()

    def predict(self, X: np.ndarray) -> np.ndarray:
        Ker = self.kernel_(X, self.x)
        Y = np.dot(self.coef_, Ker.T) + self.intercept_
        return Y
