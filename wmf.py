import numpy as np

def recompute_factors(Y,S,P,lambda_reg, dtype='float32'):
    m = S.shape[0]
    n = S.shape[1]
    f = Y.shape[1]

    X_new = np.zeros((m, f), dtype=dtype)
    for k in range(m):
        C_u = np.zeros((n, n), dtype=dtype)
        for i in range(n):
            C_u[i][i]=S[k][i]

        YCY = np.dot(np.dot(Y.T, C_u),Y)
        YCYpI = YCY + lambda_reg * np.eye(f)
        YCP=np.dot(np.dot(Y.T, C_u),P[k])
        X_new[k] = np.linalg.inv(YCYpI, YCP)

    return X_new



def factorize(S, P,num_factors, lambda_reg=1e-3, num_iterations=20, init_std=0.01, dtype='float32',
              recompute_factors=recompute_factors, *args, **kwargs):
    num_users, num_items = S.shape

    ST = S.T
    PT = P.T

    U = None  # no need to initialize U, it will be overwritten anyway
    V = np.random.randn(num_items, num_factors).astype(dtype) * init_std

    for i in range(num_iterations):
        U = recompute_factors(V, S,P, lambda_reg, dtype, *args, **kwargs)
        V = recompute_factors(U, ST,PT, lambda_reg, dtype, *args, **kwargs)

    return U, V

