import numpy as np

def pretty_print_matrix(matrix: np.ndarray, labels: list[str] | None = None) -> None:
    if labels is not None:
        header = "\t".join([""] + labels)
        print(header)
    for i, row in enumerate(matrix):
        if labels is not None:
            print("\t".join([labels[i]] + [f"{val:.4f}" for val in row]))
        else:
            print("\t".join(f"{val:.4f}" for val in row))
        