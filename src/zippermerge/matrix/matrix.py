"""
elementary.py

Functions:

rowswap(matrix, src, tgt)
    Swap row with another row.

rowscale(matrix, row, scale)
    Multiply row by the scalar.

rowreplacement(matrix, i, j, coef_i, coef_j)
    Replace row with coef_i * R_i + coef_j * R_j.

rref(matrix)
    Reduce the matrix to Reduced Row Echelon Form using the functions above.

"""

import torch

def rowswap(matrix, src, tgt):
    result = matrix.clone()
    result[[src, tgt]] = result[[tgt, src]]
    return result


def rowscale(matrix, row, scale):
    if scale == 0:
        raise ValueError("Scaling factor must be non-zero for an "
                          "elementary row operation.")
    result = matrix.clone()
    result[row] = result[row] * scale
    return result


def rowreplacement(matrix, i, j, coef_i, coef_j):
    # Use rowscale to get the two scaled rows
    scaled_i = rowscale(matrix, i, coef_i)[i]
    scaled_j = rowscale(matrix, j, coef_j)[j]

    result = matrix.clone()
    result[i] = scaled_i + scaled_j
    return result


def rref(matrix, tol=1e-10):
    M = matrix.clone().to(torch.float64)
    n_rows, n_cols = M.shape

    pivot_row = 0  # next row that needs a pivot

    for col in range(n_cols):
        if pivot_row >= n_rows:
            break  # no more rows left to turn into pivot rows

        # Partial pivoting
        sub_column = torch.abs(M[pivot_row:, col])
        best_offset = int(torch.argmax(sub_column).item())
        best_row = pivot_row + best_offset

        # If the largest candidate is zero, no pivot, move on to the next column without advancing pivot_row.
        if abs(M[best_row, col].item()) < tol:
            continue

        # Move best row into the current pivot 
        if best_row != pivot_row:
            M = rowswap(M, pivot_row, best_row)

        # Scale pivot row so pivot entry becomes 1
        pivot_val = M[pivot_row, col].item()
        M = rowscale(M, pivot_row, 1.0 / pivot_val)

        # Eliminate every other entry in this column to 0's
        for r in range(n_rows):
            if r == pivot_row:
                continue
            factor = M[r, col].item()
            if abs(factor) > tol:
                M = rowreplacement(M, r, pivot_row, 1.0, -factor)

        pivot_row += 1  # move to the next pivot row

    return M


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    A = torch.tensor([
        [1.0, 3.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 1.0, -4.0]
    ])

    print("Original matrix:")
    print(A)

    # Step 1: R1 <-> R2  (rowswap)
    step1 = rowswap(A, 0, 1)
    print(step1)

    # Step 2: (1/3) R1 (rowscale)
    step2 = rowscale(step1, 0, 1.0 / 3.0)
    print(step2)

    # Step 3: R3 = -3*R1 + R3 (rowreplacement)
    step3 = rowreplacement(step2, 2, 0, 1.0, -3.0)
    print(step3)

    # compute RREF
    print(rref(A))
