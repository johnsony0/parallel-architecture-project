"""
Negacyclic Number Theoretic Transform (NTT) implementation.

The negacyclic NTT computes polynomial evaluation at odd powers of a primitive
root. Given coefficients x[0], x[1], ..., x[N-1], the output is:

    y[k] = Σ_{n=0}^{N-1} x[n] · ψ^{(2k+1)·n}  (mod q)

where ψ is a primitive 2N-th root of unity (ψ^N ≡ -1 mod q).

This is equivalent to a cyclic NTT on "twisted" input, where each coefficient
x[n] is first multiplied by ψ^n.
"""

import jax.numpy as jnp
import jax


# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------

def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise."""
    return (a + b) % q


def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise."""
    return (a - b) % q


def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise."""
    #ai notes montgomery reduction
    return (a * b) % q


# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------


def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT.
    y[k] = sum_{n=0}^{N-1} x[n] * psi^{(2k+1)n}   (mod q)

    Args:
        x: Input coefficients, shape (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % 2N == 0
        psi_powers: Precomputed ψ^n table
        twiddles: Precomputed twiddle table

    Returns:
        jnp.ndarray: NTT output, same shape as input
    """
    batch_size, N = x.shape
    k_idx = jnp.arange(N)
    n_idx = jnp.arange(N)
    
    exponents = (2 * k_idx[:, None] + 1).astype(jnp.int64) * n_idx[None, :].astype(jnp.int64)
    index = exponents % (2 * N)
    lookup_idx = (index % N).astype(jnp.int32)
    is_negative = index >= N
    
    base_values = psi_powers[lookup_idx]
    kernel = jnp.where(is_negative, (q - base_values) % q, base_values)
    def safe_dot_product(vector_x, kernel_row):
        products = (vector_x.astype(jnp.int64) * kernel_row.astype(jnp.int64)) % q
        return jnp.sum(products) % q
    vectorized_ntt = jax.vmap(lambda p: jax.vmap(lambda k_row: safe_dot_product(p, k_row))(kernel))
    y = vectorized_ntt(x)
    return y.astype(jnp.uint32)

def prepare_tables(*, q, psi_powers, twiddles):
    """
    Optional one-time table preparation.

    Override this if you want faster modular multiplication than JAX's "%".
    For example, you can convert the provided tables into Montgomery form
    (or any other domain) once here, then run `ntt` using your mod_mul.
    This function runs before timing, so its cost is not counted as latency.
    Must return (psi_powers, twiddles) in the form expected by `ntt`.
    """
    
    return psi_powers, twiddles

