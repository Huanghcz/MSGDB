import numpy as np
from scipy.optimize import approx_fprime, minimize
from scipy.special import expit
from decimal import Decimal, getcontext

def initialize(x1, m_L, epsilon, n_max, tau):
    """
    :param x1: initial point
    :param m_L: line search parameter
    :param epsilon: stopping parameter
    :param n_max: the maximum number of null steps
    :param tau:  the stepsize tolerance
    """
    l = 1
    n_null = 0
    return x1, m_L, epsilon, n_max, tau, l, n_null


def u_update(x, y, weight, var_est, J, mL, v_i_k, i, alpha_i_j_k):
    """

    :param x: initial point
    :param y: auxiliary point
    :param weight: the weighting parameter u_i,k
    :param var_est: the variation estimate
    :param J:  bundle set
    :param mL: line search parameter
    :param v_i_k: the linear estimate
    :param i: the number of the objective function
    :param alpha_i_j_k: the linearization error
    :return: the next weighting parameter u_i,k+1
    """
    # determine next weight:
    # step a
    weight_n = weight
    u = weight
    weight_counter = 0
    weight_interp = 2 * weight * (1 - (f(y) - f(x)) / v_i_k)  # u_int
    tL = y - x
    # step b
    if np.all(tL == 0):
        p = -u * (y - x)
        var_est_p = np.linalg.norm(p) + alpha_i_j_k
        var_est_n = np.minimum(var_est, var_est_p)  # varepsilon_v^{k+1}
        # step f
        if weight_counter < - 3 and (alpha_i_j_k > np.maximum(var_est_n, -10 * v_i_k)):
            u = weight_interp
        weight_n = min(u, 10 * weight)
        weight_counter_n = min(weight_counter - 1, -1)
        if weight_n != weight:
            weight_counter_n = - 1
        # step c
    elif np.all(tL > 0):
        var_est_n = min(var_est, -2 * np.max(v_i_k))
        if weight_counter > 0:
            if alpha_i_j_k <= mL * v_i_k:
                u = weight_interp
            # step d
            if weight_counter > 3:
                u = weight / 2
        # step e
        weight_n = np.maximum(u, weight / 10, np.array([1e-10]))
        weight_counter_n = np.maximum(weight_counter + 1, 1)
        if np.all(weight_n != weight):
            weight_counter_n = 1
    return float(weight_n)


def direction_search(x, f, grad_f,i,var_est):
    """

    :param x: initial point
    :param f: the objective function
    :param grad_f: the gradient of f
    :param i: the number of f
    :param var_est: the variation estimate
    :return: d: the descent direction
    """
    # Initialize variables
    di = []
    # Step A: Initialization
    u_i = np.array([1])
    y_i = x.copy()
    J_i = {1}
    d_i_k = 0
    xi_i = np.array([grad_f(y_i)])
    k = 1

    while True:
        # Step B: Direction search
        alpha_i_j_k = np.array([f(x) - f(y_i) - np.dot(xi_i[-1], x - y_i)])
        u_now = u_i[-1]

        d_i_k += -1 / u_now * np.sum([u_j * xi_j for xi_j, u_j in zip(xi_i, u_i)], axis=0)

        sum_term = np.sum([u_j * xi_j for xi_j, u_j in zip(xi_i, u_i)], axis=0)
        square_term = np.linalg.norm(sum_term) ** 2
        second_term = np.sum(np.fromiter((u_j * alpha_i_j_k_j for alpha_i_j_k_j, u_j in zip(alpha_i_j_k, u_i)), float))

        v_i_k = float(-((1 / u_now) * square_term + second_term))

        y_i = x + d_i_k

        # print(f't1={f(y_i)},t2={f(x) + m_L * v_i_k}')

        if f(y_i) <= f(x) + m_L * v_i_k:
            # Descent condition satisfied
            di.append(d_i_k)
            break
        else:
            # Step C: Update
            J_i = J_i.union({k + 1}) if k < 7 else J_i.union({k + 1}) - {k - 7}
            xi_i_next = grad_f(y_i)
            if k < 7:
                xi_i = np.vstack([xi_i, xi_i_next])
            else:
                xi_i = np.vstack([xi_i, xi_i_next])
                xi_i = xi_i[1:]
            # u_next = u_update(x, y_i, u_now, var_est, J_i, mL=0.25, v_i_k=v_i_k, i=i, alpha_i_j_k=alpha_i_j_k)

            u_next = 5 * u_now
            # print(f'u_next={u_next}')
            if k < 7:
                u_i = np.vstack([u_i, u_next])
            else:
                u_i = np.vstack([u_i, u_next])
                u_i = u_i[1:]
            k += 1


    return di


def search_direction_candidate(A,B):
    """

    :param A: the list of the gradient
    :param B: the list of the linearization error
    :return: d: the candidate descent direction
    """

    m = len(A)
    def objective_function(lambda_vector):
        term1 = np.sum(lambda_vector.reshape(m) * A, axis=(0, 1))
        term2 = np.sum(lambda_vector.reshape(m) * B)
        return (np.linalg.norm(term1)) ** 2 + term2

    def constraint(lambda_vector):
        return np.sum(lambda_vector) - 1.0

    constraints = [{'type': 'eq', 'fun': constraint}]


    initial_guess = np.ones(m) / m


    result = minimize(objective_function, initial_guess, constraints=constraints)


    optimal_lambda = result.x.reshape(m)
    res = np.sum(optimal_lambda * A)

    return res


def armijo_line_search(func, gradient, direction, x, alpha=1.0, beta=0.1, c=0.1, min_step=0.001, max_iterations=10000, tol=1e-6):
    """
    Armijo line search algorithm, returns only the step length

    Parameters:
    - func: Objective function
    - gradient: Gradient of the objective function
    - x: Current search point
    - alpha: Initial step length
    - beta: Scaling factor
    - c: Constant in the Armijo condition
    - min_step: Minimum value for the step length
    - max_iterations: Maximum number of iterations
    - tol: Convergence tolerance

    Returns:
    - alpha: Step length obtained from the Armijo line search
    """

    iterations = 0
    while iterations < max_iterations:
        f_x = func(x)
        g_x = gradient(x)

        while np.all(func(x + alpha * direction) > f_x + c * alpha * np.dot(g_x, direction)):
            print(f'turn')
            alpha *= beta

        if np.linalg.norm(alpha * direction) <= tol and alpha <= min_step:
            print(f'panduan')
            break

        iterations += 1

    return max(alpha, min_step)

def solve_dual(d1, d2):
    """

    :param d1: the descent direction of f1
    :param d2: the descent direction of f2
    :return: d: the descent direction of f
    """
    def objective(w, d1, d2):
        res = (np.linalg.norm(w * d1 + (1 - w) * d2)) ** 2
        return res

    initial_guess = 0.5
    result = minimize(objective, initial_guess, args=(d1, d2))
    optimal_w = result.x[0]
    res = optimal_w*d1 + (1-optimal_w)*d2
    return res


def main_algorithm(x1, m_L, epsilon, n_max, tau, f, df, m):
    """

    :param x1: initial point
    :param m_L: the line search parameter
    :param epsilon: the stopping parameter
    :param n_max: the null step
    :param tau: the stepsize tolerance
    :param f: the objective function
    :param df: the gradient of f
    :param m:
    :return: x: the optimal point
    """
    x, m_L, epsilon, n_max, tau, l, n_null = initialize(x1, m_L, epsilon, n_max, tau)
    var_est = float('inf')
    A = []
    B = []
    terminate_outer_loop = False  # Flag to control the outer loop

    while not terminate_outer_loop:

        # print(f'loop')
        # step2 Direction finding
        print(f'now x={x}')
        print(f'now func value={np.array([f1(x),f2(x)])}')
        d1_value = direction_search(x, f1, grad_f1, 1, var_est)
        print(f'tt')
        d1 = np.array(d1_value[0].tolist())
        # print(f'd1={d1}')
        d2_value = direction_search(x, f2, grad_f2, 2, var_est)
        d2 = np.array(d2_value[0].tolist())
        # print(f'd2={d2}')
        # step 3: Search direction candidate finding
        d = -(solve_dual(d1, d2))
        # print(f'd={d}')
        # step 4: Stopping criterion and descent test
        term = np.linalg.norm(d)
        print(f'term={np.linalg.norm(d)}')
        while np.linalg.norm(d) > epsilon:
            # print(f'success to loop')
            # step 7: Line search
            if f(x + tau * d)[0] <= f(x)[0] and f(x + tau * d)[1] <= f(x)[1]:
                # print(f'success to loop2')
                t = armijo_line_search(f, subgradient, d, x, alpha=1.0, beta=0.1, c=0.1, min_step=tau,
                                       max_iterations=100, tol=1e-6)
                # print(f't={t}')
                x_old = x
                x = x.astype(float)
                x += t * d
                l += 1
                # if np.all(f(x_old) - f(x) < 1e-2):
                if np.linalg.norm(d) <= 0.5:
                    terminate_outer_loop = True
                else:
                    print(f'now x={x},x_old={x_old}')
                    terminate_outer_loop = False
                # terminate_outer_loop = False
                n_null = 0  # Set the flag to terminate the outer loop
                break
            else:
                # step 5: Null step
                # print(f'success to loop3')
                xi = subgradient(x + d)
                vector1 = xi[0]
                vector2 = xi[1]
                A.append(vector1)
                A.append(vector2)
                # print(f'A={A}')
                alpha_i_j = np.array([
                    f(x)[0] - f(x + d)[0] - np.dot(xi[0], (x - (x + d)).flatten()),
                    f(x)[1] - f(x + d)[1] - np.dot(xi[1], (x - (x + d)).flatten())
                ])
                term1 = alpha_i_j[0]
                term2 = alpha_i_j[1]
                B.append(term1)
                B.append(term2)
                n_null += 1
                # Step 6: (Search direction candidate finding)
                if n_null > n_max:
                    # print(f'success to loop4')
                    w = search_direction_candidate(xi, alpha_i_j)
                    # print(f'w={w}')
                    d = -(w * xi[0] + (1 - w) * xi[1])
                    # print(f'd22={d}')
                    n_null = 0  # Reset n_null to 0 to re-enter Step 4
                    # No need to set terminate_outer_loop, which will continue the outer loop
                    # Turn to step 4
                else:
                    # print(f'need to reloop')
                    t = armijo_line_search(f, subgradient, d, x, alpha=1.0, beta=0.01, c=0.1, min_step=tau,
                                           max_iterations=100, tol=1e-6)
                    x = x + t*d
                    terminate_outer_loop = False  # Set the flag to terminate the outer loop
                    break

                # Turn to step 2B
    return x, l



def f1(x):
    getcontext().prec = 9

    # Convert inputs to Decimal type
    x_decimal = [Decimal(str(val)) for val in x]

    a = (x_decimal[0] ** 4 + x_decimal[1] ** 2)
    b = ((2 - x_decimal[0]) ** 2 + (2 - x_decimal[1]) ** 2)
    c = (2 * expit(float(x_decimal[1]) - float(x_decimal[0])))


    res = max(float(a), float(b), float(c))

    return res

# def f2(x):
#     a = (5 * x[0] + x[1])
#     b = (-5 * x[0] + x[1])
#     c = (x[0] ** 2 + x[1] ** 2 + 4 * x[0])
#     res = np.max([a,b,c])
#     return res
def f2(x):
    a = (-x[0] - x[1])
    b = (-x[0] - x[1] + x[0]**2 + x[1]**2 -1)
    res = np.max([a,b])
    return res

# def f1(x):
#     a = (x[0] ** 2 + x[1] ** 2)
#     b = ((2 - x[0]) ** 2 + (2 - x[1]) ** 2)
#     c = (2 * expit(x[1] - x[0]))
#     res = np.max([a,b,c])
#     return res
#
# def f2(x):
#     a = (-x[0] - x[1])
#     b = (-x[0] - x[1] + x[0]**2 + x[1]**2 -1)
#     res = np.max([a,b])
#     return res

def f(x):
    res = np.array([f1(x), f2(x)])
    return res



def grad_f1(x):
    def f1_wrapper(x):
        return f1(x)

    epsilon = np.sqrt(np.finfo(float).eps) * 10
    grad = approx_fprime(x.flatten(), f1_wrapper, epsilon)
    return grad

def grad_f2(x):
    def f2_wrapper(x):
        return f2(x)

    epsilon = np.sqrt(np.finfo(float).eps) * 10
    grad = approx_fprime(x.flatten(), f2_wrapper, epsilon)
    return grad



def subgradient(x):
    res = np.array([grad_f1(x), grad_f2(x)])
    return res


# Example usage:
# Set initial parameters
x1 = np.array([2, 2])
m_L = 0.25
epsilon = 1e-9
n_max = 2
tau = 0.001
m = 2  # Number of functions

result = main_algorithm(x1, m_L, epsilon, n_max, tau, f, subgradient, m)
print("Optimal solution:",result[0],"Optimal value:", f(result[0]),f'Iteration:{result[1]}')
