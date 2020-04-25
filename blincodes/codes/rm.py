"""Module for working with binary Reed-Muller codes."""

from blincodes import vector, matrix


def generator(param_r, param_m):
    """Make Reed-Muller RM(r,m) generator matrix."""
    param_m = max(0, param_m)
    monoms = []
    for i, j in ((1 << (param_m - p - 1), 1 << p) for p in range(param_m)):
        monoms.append(vector.Vector(int(('0' * i + '1' * i) * j, 2),
                                    1 << param_m))
    gen_matrix = [vector.Vector(int('1' * (1 << param_m), 2), 1 << param_m)]
    for i in range(1, param_r + 1):
        if i == 1:
            curr_layer = [(i, monoms[i]) for i in range(len(monoms))]
            gen_matrix += [monoms[i] for i in range(len(monoms))]
            continue
        next_layer = []
        for max_monom, row in curr_layer:
            for j in range(max_monom + 1, param_m):
                next_layer.append((j, row * monoms[j]))
                gen_matrix.append(next_layer[-1][1])
        curr_layer = next_layer
    return matrix.Matrix(
        (row.value for row in gen_matrix),
        1 << param_m)


def parity_check(param_r, param_m):
    """Make Reed-Muller RM(r,m) parity check matrix."""
    return generator(param_m - param_r - 1, param_m)
