import json
from matplotlib import pyplot as plt
from intvalpy import Interval, Tol, precision
from intvalpy_fix import IntLinIncR2

precision.extendedPrecisionQ = True

def load_data(directory, side):
    values_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    loaded_data = []
    for i in range(8):
        loaded_data.append([])
        for j in range(1024):
            loaded_data[i].append([(values_x[i // 100], 0) for i in range(100 * len(values_x))])

    for offset, value_x in enumerate(values_x):
        data = {}
        with open(directory + "/" + str(value_x) + "lvl_side_" + side + "_fast_data.json", "rt") as f:
            data = json.load(f)
        for i in range(8):
            for j in range(1024):
                for k in range(len(data["sensors"][i][j])):
                    loaded_data[i][j][offset * 100 + k] = (value_x, data["sensors"][i][j][k])

    return loaded_data

def plot_data(points, b_data_1, rads_1, b_data_2):
    x, y = zip(*points)
    plt.figure()
    plt.title("Y(x)")
    plt.scatter(x, y, label="medians")
    plt.plot([-0.5, 0.5], [b_data_1[1] + b_data_1[0] * -0.5, b_data_1[1] + b_data_1[0] * 0.5], label="Method 1")
    plt.plot([-0.5, 0.5], [b_data_2[1] + b_data_2[0] * -0.5, b_data_2[1] + b_data_2[0] * 0.5], label="Method 2")
    plt.legend()

    plt.figure()
    plt.title("Y(x) - b_1*x - b0 method 1")
    for i in range(len(y)):
        plt.plot([i, i], [y[i] - rads_1[i] - b_data_1[1] - b_data_1[0] * x[i], y[i] + rads_1[i] - b_data_1[1] - b_data_1[0] * x[i]], color="k", zorder=1)
    plt.scatter([i for i in range(len(y))], [y[i] - b_data_1[1] - b_data_1[0] * x[i] for i in range(len(y))], label="medians", zorder=2)
    plt.legend()


# using Tol
def regression_type_1(points):
    x, y = zip(*points)
    # build intervals out of given points
    weights = [1 / 16384] * len(y)
    # we know that y_i = b_0 + b_1 * x_i
    # or, in other words
    # Y = X * b, where X is a matrix with row (x_i, 1), and b is a vector (b_1, b_0)
    X_mat = Interval([[[x_el, x_el], [1, 1]] for x_el in x])
    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    # find argmax for Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([
                [[x[i], x[i]], [1, 1]]
            ])
            Y_vec_small = Interval([[y[i], weights[i]]], midRadQ=True)
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                weights[i] = abs(y[i] - (x[i] * b_vec[0] + b_vec[1])) + 1e-8

    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    # find argmax for Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)
    print(tol_val)
    print(b_vec)
    '''
    vertices1 = IntLinIncR2(X_mat, Y_vec, show=False)
    vertices2 = IntLinIncR2(X_mat, Y_vec, consistency='tol', show=False)

    plt.figure()
    for v in vertices1:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            plt.fill(x, y, linestyle='-', linewidth=1, color='gray', alpha=0.5)
            plt.scatter(x, y, s=0, color='black', alpha=1)

    for v in vertices2:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            plt.fill(x, y, linestyle='-', linewidth=1, color='blue', alpha=0.3)
            plt.scatter(x, y, s=10, color='black', alpha=1)
    '''
    return b_vec, weights

# using twin arithmetics
def regression_type_2(points):
    x, y = zip(*points)
    eps = 1 / 16384

    # first of all, lets build y_ex and y_in
    x_new = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y_ex_up = [-float('inf')] * 11
    y_ex_down = [float('inf')] * 11
    y_in_up = [-float('inf')] * 11
    y_in_down = [float('inf')] * 11
    for i in range(len(x)):
        index = i // 100
        y_ex_up[index] = max(y_ex_up[index], y[i] + eps)
        y_ex_down[index] = min(y_ex_down[index], y[i] - eps)
    for i in range(len(x_new)):
        y_list = list(y[i * 100 : (i + 1) * 100])
        y_list.sort()
        y_in_down[i] = y_list[25] - eps
        y_in_up[i] = y_list[75] + eps

    X_mat = []
    Y_vec = []
    for i in range(len(x_new)):
        x_el = x_new[i]
        # y_ex_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_ex_up[i]])
        # y_in_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_in_up[i]])
        # y_ex_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_ex_up[i]])
        # y_in_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])

    # now we have matrix X * b = Y, but with some "additional" rows
    # we can walk over all rows and if some of them is less than 0, we can just remove it at all
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)
    if tol_val < 0:
        to_remove = []
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([X_mat[i]])
            Y_vec_small = Interval([Y_vec[i]])
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            del X_mat[i]
            del Y_vec[i]

    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)

    vertices1 = IntLinIncR2(X_mat_interval, Y_vec_interval)
    vertices2 = IntLinIncR2(X_mat_interval, Y_vec_interval, consistency='tol')

    plt.figure()
    plt.title("Uni and Tol method 2")
    plt.xlabel("b0")
    plt.ylabel("b1")
    for v in vertices1:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            plt.fill(x, y, linestyle='-', linewidth=1, color='gray', alpha=0.5, label="Uni")
            plt.scatter(x, y, s=0, color='black', alpha=1)

    for v in vertices2:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            plt.fill(x, y, linestyle='-', linewidth=1, color='blue', alpha=0.3, label="Tol")
            plt.scatter(x, y, s=10, color='black', alpha=1)
            plt.scatter([b_vec[0]], [b_vec[1]], s=10, color='red', alpha=1, label="argmax Tol")
    plt.legend()
    return b_vec

if __name__ == "__main__":
    side_a_1 = load_data("bin/04_10_2024_070_068", "a")
    b_vec2 = regression_type_2(side_a_1[0][0])
    b_vec, rads = regression_type_1(side_a_1[0][0])
    plot_data(side_a_1[0][0], b_vec, rads, b_vec2)
    plt.show()
    print("hello")