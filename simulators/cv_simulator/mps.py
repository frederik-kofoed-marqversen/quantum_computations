import numpy as np
from functools import reduce
import inspect


def randomized_range_finder(A: np.ndarray, l: int, q: int, *, rng_seed: int=None) -> np.ndarray:
    # From: N. Halko et al. Finding structure with randomness: Probabilistic algorithms for 
    # constructing approximate matrix decompositions. Dec. 14, 2010. doi: 10.48550/arXiv.0909.4061. 
    # arXiv: 0909.4061[math].
    
    # For nxm matrix A
    # Find mxl matrix Q such that
    # Q @ Q^† @ A ≈ A
    rng = np.random.default_rng(rng_seed)
    O = rng.normal(0, 1, size=(A.shape[1], l))
    Y = A @ O
    Q, _ = np.linalg.qr(Y, 'reduced')
    for _ in range(q):
        Y = A.T.conj() @ Q
        Q, _ = np.linalg.qr(Y, 'reduced')
        Y = A @ Q
        Q, _ = np.linalg.qr(Y, 'reduced')
    return Q

def randomized_truncated_svd(A: np.ndarray, k: int, *, rng_seed: int=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Inspired by scikit-learn implementation
    
    p = 10  # fixed oversampling parameter
    q = 7 if k < 0.1 * min(A.shape) else 4  # auto power iterations as in scikit-learn
    
    # Much more efficient to reduce the larger dimension
    transpose = (A.shape[0] < A.shape[1])
    if transpose:
        A = A.T
    
    Q = randomized_range_finder(A, k + p, q, rng_seed=rng_seed)
    # Now A ≈ Q @ Q^† @ A. Compute SVD of reduced matrix
    # Q^† @ A = B = U @ S @ Vh
    # Such that approximate SVD for A is
    # A ≈ Q @ Q^† @ A = Q @ B = Q @ U @ S @ Vh = (Q @ U) @ S @ Vh
    B = Q.T.conj() @ A
    U, S, Vh = np.linalg.svd(B)

    # Keep only k largest singular values
    U, S, Vh = Q @ U[:, :k], S[:k], Vh[:k, :]

    if transpose:
        return Vh.T, S, U.T
    else:
        return U, S, Vh

def tensor_svd(
    tensor: np.ndarray,
    left_indices: list[int],
    right_indices: list[int],
    *,
    max_bond_dim: int = np.inf,
    abs_err: float = 0,
    rel_err: float = 1e-12,
    rng_seed: int=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits rank n tensor by SVD.
    Returns tensors m1 and m2 such that they share one internal edge j, and such that m1 owns the 
        edges corresponding to left_indices+[j] and m2 those of [j]+right_indices in that order.
    While truncating, max_singular_values takes priority, after which the larger of abs_truncation_err and 
        sum(singular values) * rel_truncation_err is used as upper threshold for the sum of removed values
    Reconstructing tensor (up to approximation) can be done as:
        tensor ~= np.einsum("left_indices j, j right_indices -> list[range(n)]", m1, m2)"""

    if not sorted(left_indices + right_indices) == list(range(len(tensor.shape))):
        raise IndexError("Output indices does not match indices of initial tensor")
    
    # perform svd on the flattened tensor `m`
    m = tensor
    m = np.moveaxis(m, left_indices + right_indices, range(len(tensor.shape)))
    m = np.reshape(m, (np.prod([tensor.shape[i] for i in left_indices]), np.prod([tensor.shape[i] for i in right_indices])))

    # svd returns singular values sorted in increasing order
    full_rank = min(*m.shape)
    if max_bond_dim * 10 < full_rank:
        u, s, vh = randomized_truncated_svd(m, max_bond_dim, rng_seed=rng_seed)
    else:
        u, s, vh = np.linalg.svd(m, full_matrices=False)  # Exact truncated SVD is very expensive
    sqrt_s = np.sqrt(s)
    
    # compute number of singular values `r` to keep
    allowed_truncation_err = max(0, abs_err, sum(s) * rel_err)
    r = sum(np.flip(s).cumsum() > allowed_truncation_err)
    r = min(r, len(s), max(0, max_bond_dim))
    
    # compute the two resulting tensors
    m1 = np.einsum("ij,j -> ij", u[:, :r], sqrt_s[:r])
    m2 = np.einsum("i,ij -> ij", sqrt_s[:r], vh[:r, :])
    m1 = np.reshape(m1, [tensor.shape[i] for i in left_indices] + [r])
    m2 = np.reshape(m2, [r] + [tensor.shape[i] for i in right_indices])
    
    return m1, m2

SVD_OPTIONS = {name: param for name, param in inspect.signature(tensor_svd).parameters.items() 
               if param.kind == inspect.Parameter.KEYWORD_ONLY}

class MPS:
    def __init__(self, domain: np.ndarray, tensors: list[np.ndarray]):
        # Add trivial edges (tensor product) if tensor is a vector
        self.tensors: list[np.ndarray] = [t.reshape(1, -1, 1) if t.ndim == 1 else t for t in tensors]
        self.domain: np.ndarray = domain
        self.diff: float = abs(domain[-1] - domain[0]) / (len(domain) - 1)
        self.validate()
    
    def __getitem__(self, index):
        return self.tensors[index]

    def __setitem__(self, index, value):
        self.tensors[index] = value
    
    def __len__(self):
        return len(self.tensors)
    
    def __iter__(self):
        return self.tensors.__iter__()
    
    def copy(self):
        return MPS(
            domain=self.domain.copy(),
            tensors=[t.copy() for t in self.tensors]
        )
    
    def shape(self):
        return tuple(t.shape for t in self.tensors)
    
    def validate(self):
        if not isinstance(self.domain, np.ndarray) or self.domain.ndim != 1:
            raise TypeError("Domain must be a 1D numpy array.")
        # Currently domain must be a sorted list of equidistant points.
        if not np.allclose(np.diff(self.domain, 2), 0, atol=np.finfo(self.domain.dtype).eps**0.5):
            raise ValueError("Domain is not an arithmetic progression.")
        if not np.isclose(self.diff, abs(self.domain[-1] - self.domain[0]) / (len(self.domain) - 1)):
            raise ValueError("Stored difference does not match current domain.")
        
        if len(self.tensors) == 0:
            # Empty MPS is an allowed edge case
            return
        
        for idx, tensor in enumerate(self.tensors):
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Tensor at index {idx} is not a numpy array.")
            if tensor.ndim != 3:
                raise ValueError(f"Tensor at index {idx} does not have exactly three axes.")
            # Currently all tensors must share physical dimensions as defined by the given domain.
            if tensor.shape[1] != len(self.domain):
                raise ValueError(f"Tensor at index {idx} does not have the right physical dimension.")
    
        # Currently MPS must be linearand so have trivial edges in either end.
        if self.tensors[0].shape[0] != 1:
            raise ValueError("Left-most tensor does not have a trivial left edge")
        if self.tensors[-1].shape[2] != 1:
            raise ValueError("Right-most tensor does not have a trivial right edge")
        
        for idx, (t1, t2) in enumerate(zip(self.tensors, self.tensors[1:])):
            if t1.shape[2] != t2.shape[0]:
                raise ValueError(f"Tensors at indices {idx} and {idx+1} do not have compatible bond dimensions.")
    
    def contract(self) -> np.ndarray:
        return np.squeeze(reduce(lambda t1, t2: np.tensordot(t1, t2, axes=1), self.tensors))
    
    def norm(self) -> float:
        res = reduce(lambda res, t: np.einsum("ab,aci,bcj -> ij", res, t, np.conj(t), optimize=True), self.tensors, np.ones((1, 1)))
        res = res[0, 0] * self.diff**len(self.tensors)
        res = np.sqrt(np.real_if_close(res))
        return res
    
    def density_mps(self) -> np.ndarray:
        """Computes the density MPS which is an MPS where each node has two physical dimensions"""
        raise NotImplementedError("Not yet implemented")
    
    def partial_density_mps(self, axis: int) -> np.ndarray:
        """Computes the partial trace of the density matrix such that only the subsystem 
        corresponding to axis is left."""

        if axis < 0 or axis >= len(self.tensors):
            raise IndexError(f"axis={axis} out of bounds")
        
        left_tensors = [np.ones((1, 1))] + self.tensors[:axis]
        left  = reduce(lambda res, t: np.einsum("ab,aci,bcj -> ij", res, t, np.conj(t), optimize=True), left_tensors)
        right_tensors = self.tensors[axis+1:] + [np.ones((1, 1))]
        right = reduce(lambda res, t: np.einsum("ica,jcb,ab -> ij", t, np.conj(t), res, optimize=True), right_tensors[::-1])
        
        t = self.tensors[axis]
        result = np.einsum("ab,aic,bjd,cd -> ij", left, t, np.conj(t), right, optimize=True)
        return result * self.diff**(len(self.tensors) - 1)
    
    @staticmethod
    def fidelity(a: 'MPS', b: 'MPS') -> float:
        # Currently assuming that `a` and `b` have the same domain!!!
        dq = a.diff
        res = np.ones((1, 1))
        for m1, m2 in zip(a.tensors, b.tensors, strict=True):
            res = np.einsum("ab,aci,bcj -> ij", res, m1, np.conj(m1), optimize=True)
        res = res[0, 0] * dq**len(a)
        res = np.abs(res)**2
        return res