"""
Binary IVM implementation 
"""

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
from tqdm import tqdm


class IVMBinary(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        alpha: float = 1e-3,
        tol: float = 1e-3,
        std: float = 10
    ) -> None:
        """
        Args:
            alpha (float, optional): Regularization strength. Defaults to 1.
            tol (float, optional): Tolerance. Defaults to 1e-3.
            std (float, optional): RBF standard deviation. Defaults to 0.1.
        """

        self.alpha = alpha
        self.tol = tol
        self.std = std

    def f_(self, X1: np.array, X2: np.array, a: np.array):
        return (rbf_kernel(X1, X2, gamma=self.gamma_) @ a).reshape(-1, 1)

    def fit(self, X: np.array, y: np.array):
        X, y = check_X_y(X=X, y=y)
        y = y.reshape(-1, 1)
        n = len(X)

        # At train time, store only indices, after - import points with corresponding weights
        S = []
        a = np.ones((n, 1)) / n
        self.gamma_ = 1 / (2 * self.std ** 2)

        # Temporary variables, not needed after
        K1 = np.empty((n, 0))
        K2 = np.empty((0, 0))
        ones = np.ones((n, 1))
        H = []
        # Worst case: all points are import
        n_iterations = 0
        while len(S) < n:
            logger.info(f'Iteration #{n_iterations + 1}')
            hist = []
            for i, x_curr in tqdm(enumerate(X), total=len(X)):
                if i in S:
                    continue
                S_tmp = S + [i]

                # Compute distances
                d = rbf_kernel(X, x_curr.reshape(1, -1), gamma=self.gamma_)

                # Temporary K1 and K2
                K1_curr = np.hstack([K1, d])
                K2_curr = np.pad(K2, (0, 1), constant_values=(0, 0))
                K2_curr[-1] = d[S_tmp].reshape(-1,)
                K2_curr[:, -1] = d[S_tmp].reshape(-1,)

                # NLL
                main_loss = (1 / n) * ones.T @ np.log2(ones + np.exp(-y * (K1_curr @ a[S_tmp])))
                reg_loss = (self.alpha / 2) * a[S_tmp].T @ K2_curr @ a[S_tmp]
                H_curr = float(main_loss + reg_loss)

                hist.append((i, H_curr, d))

            # Select best point
            hist.sort(key=lambda x: (x[1], x[0]))
            argmin_idx, min_loss, d = hist[0]
            logger.info(f'loss: {min_loss}, import point idx: {argmin_idx}')
            S.append(argmin_idx)
            H.append(min_loss)

            # Update K1 and K2
            K1 = np.hstack([K1, d])
            K2 = np.pad(K2, (0, 1), constant_values=(0, 0))
            K2[-1] = d[S].reshape(-1,)
            K2[:, -1] = d[S].reshape(-1,)

            # Update weights
            p = 1 / (1 + np.exp(-y * self.f_(X, X[S], a[S])))
            W = np.diag((p * (1 - p)).reshape(-1,))
            z = (1 / n) * (K1 @ a[S] + np.linalg.inv(W) @ (y * p))
            a[S] = np.linalg.inv(
                ((1 / n) * K1.T @ W @ K1 + self.alpha * K2)
            ) @ K1.T @ W @ z

            # Stopping criteria
            if len(H) >= 2:
                eps = abs(H[-1] - H[-2]) / abs(H[-1])
                if eps <= self.tol:
                    logger.info('Reached stopping criteria')
                    break
            n_iterations += 1

        # Save import points with their weights
        self.a_ = a[S]
        self.X_ = X[S]
        logger.info(f'Import points set size: {len(self.X_)}')

        return self

    def predict(self, X: np.array):
        return self.f_(X, self.X_, self.a_).reshape(-1,)


if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=100,
        random_state=42
    )

    clf = IVMBinary()
    clf.fit(X_train, y_train)

    preds_test = clf.predict(X_test)
    test_rocauc = roc_auc_score(y_true=y_test, y_score=preds_test)
    logger.info(f'Test ROC-AUC: {test_rocauc}')
    logger.info(classification_report(
        y_true=y_test,
        y_pred=(preds_test >= 0.5) * 1
    ))
