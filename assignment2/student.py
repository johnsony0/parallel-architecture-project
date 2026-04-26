"""
Assignment 2 student implementation reference skeleton.

This file documents the frozen student-facing API.
Only 32-bit kernels are compulsory in the base track.
64-bit and 128-bit kernels are intentionally left unimplemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    return ((jnp.uint64(a) + jnp.uint64(b)) % jnp.uint64(q)).astype(jnp.uint32)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    return ((jnp.uint64(q) + jnp.uint64(a) - jnp.uint64(b)) % jnp.uint64(q)).astype(jnp.uint32)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    return ((jnp.uint64(a) * jnp.uint64(b)) % jnp.uint64(q)).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    """Optional 64-bit modular add kernel."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def mod_sub_64(a, b, q):
    """Optional 64-bit modular subtract kernel."""
    raise NotImplementedError


def mod_mul_64(a, b, q):
    """Optional 64-bit modular multiply kernel."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_128(a, b, q):
    """Optional 128-bit modular add kernel."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def mod_sub_128(a, b, q):
    """Optional 128-bit modular subtract kernel."""
    raise NotImplementedError


def mod_mul_128(a, b, q):
    """Optional 128-bit modular multiply kernel."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Frozen dispatch API
# -----------------------------------------------------------------------------

def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    if int(bit_width) == 64:
        return mod_sub_64(a, b, q)
    if int(bit_width) == 128:
        return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    if int(bit_width) == 64:
        return mod_mul_64(a, b, q)
    if int(bit_width) == 128:
        return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """Compulsory 32-bit MLE update."""
    diff = mod_sub(one_eval, zero_eval, q)
    prod = mod_mul(diff, target_eval, q)
    return mod_add(prod, zero_eval, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """Optional 64-bit MLE update."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """Optional 128-bit MLE update."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")
    
from functools import partial

def compute_composition_indexed(expr_indices, extensions, q):
    """
    Computes composition using indices instead of names.
    Inlined into JIT kernels; expr_indices must be a static tuple of tuples.
    """
    total_sum = jnp.zeros_like(extensions[0], dtype=jnp.uint32)
    for term_indices in expr_indices:
        product = jnp.ones_like(extensions[0], dtype=jnp.uint32)
        for var_idx in term_indices:
            val = extensions[var_idx]
            product = mod_mul_32(product, val, q)
        total_sum = mod_add_32(total_sum, product, q)
    return total_sum

@partial(jax.jit, static_argnums=(3,))
def compute_round_points_kernel(tables_tuple, x_points, q, expr_indices):
    """
    JIT-optimized kernel to compute all points for a round polynomial.
    """
    def eval_at_x(x):
        extensions = []
        for table in tables_tuple:
            evens = table[0::2]
            odds = table[1::2]
            extensions.append(mle_update_32(evens, odds, x, q=q))
        
        pair_compositions = compute_composition_indexed(expr_indices, extensions, q)
        return (jnp.sum(pair_compositions.astype(jnp.uint64)) % jnp.uint64(q)).astype(jnp.uint32)

    return jax.vmap(eval_at_x)(x_points)

@jax.jit
def fold_tables_kernel(tables_tuple, challenge, q):
    """Vectorized folding of all tables for the next round."""
    new_tables = []
    for table in tables_tuple:
        evens = table[0::2]
        odds = table[1::2]
        new_tables.append(mle_update_32(evens, odds, challenge, q=q))
    return tuple(new_tables)

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    GPU-optimized 32-bit sumcheck using JAX.
    """
    degree = max(len(term) for term in expression)
    q = jnp.asarray(q, dtype=jnp.uint32)
    challenges = jnp.asarray(challenges, dtype=jnp.uint32)
    x_points = jnp.arange(degree + 1, dtype=jnp.uint32)
    
    # 1. Pre-index the expression for JIT compatibility
    standard_names = ("a", "b", "c", "d", "e", "g")
    active_names = [n for n in standard_names if n in eval_tables]
    # Fallback for any unexpected names not in standard list
    for n in eval_tables:
        if n not in active_names:
            active_names.append(n)
            
    name_to_idx = {name: i for i, name in enumerate(active_names)}
    expr_indices = tuple(tuple(name_to_idx[var] for var in term) for term in expression)
    
    # 2. Prepare initial tables as a tuple of JAX arrays
    current_tables = tuple(jnp.asarray(eval_tables[name], dtype=jnp.uint32) for name in active_names)
    
    all_round_evals = []

    for r in range(num_rounds):
        # 3. Compute all points for this round's univariate polynomial in one kernel call
        round_poly = compute_round_points_kernel(current_tables, x_points, q, expr_indices)
        all_round_evals.append(round_poly)

        # 4. Fold tables for the next round (except the last round)
        if r < num_rounds - 1:
            current_tables = fold_tables_kernel(current_tables, challenges[r], q)

    # Round 0 evaluations give the initial sum: g_1(0) + g_1(1)
    claim0 = mod_add_32(all_round_evals[0][0], all_round_evals[0][1], q)

    return jnp.asarray(claim0, dtype=jnp.uint32), jnp.stack(all_round_evals)



def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 64-bit sumcheck path."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 128-bit sumcheck path."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")
