from scipy.optimize import minimize, rosen


def scipy_minimize():
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
    print(res)
    print(res.x)

if __name__ == '__main__':
    scipy_minimize()
