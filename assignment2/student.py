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
    res_plus = a + b
    res_minus = a - (q - b)
    return jnp.where(a >= q - b, res_minus, res_plus)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    res_minus = a - b
    res_plus = (q - b) + a
    return jnp.where(a < b, res_plus, res_minus)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    return ((jnp.uint64(a) * jnp.uint64(b)) % jnp.uint64(q)).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    """Optional 64-bit modular add kernel."""
    res_plus = a + b
    res_minus = a - (q - b)
    return jnp.where(a >= q - b, res_minus, res_plus)


def mod_sub_64(a, b, q):
    """Optional 64-bit modular subtract kernel."""
    res_minus = a - b
    res_plus = (q - b) + a
    return jnp.where(a < b, res_plus, res_minus)


def mod_mul_64(a, b, q):
    """64-bit modular multiply kernel using bit-by-bit approach."""
    a = jnp.uint64(a)
    b = jnp.uint64(b)
    q = jnp.uint64(q)

    # Initialize carry with the broadcasted shape of a and b to support vectorized ops.
    init_carry = jnp.zeros_like(a * b)

    def body_fun(i, carry):
        res = mod_add_64(carry, carry, q)
        bit = (a >> (63 - i)) & jnp.uint64(1)
        term = jnp.where(bit, b, jnp.uint64(0))
        return mod_add_64(res, term, q)

    return jax.lax.fori_loop(0, 64, body_fun, init_carry)


def mod_sum_64(arr, q):
    """Modular sum of an array using 64-bit add."""

    def body(carry, x):
        return mod_add_64(carry, x, q), None

    res, _ = jax.lax.scan(body, jnp.uint64(0), arr)
    return res


def compute_composition_indexed_64(expr_indices, extensions, q):
    """Computes composition for 64-bit track."""
    total_sum = jnp.zeros_like(extensions[0], dtype=jnp.uint64)
    for term_indices in expr_indices:
        product = jnp.ones_like(extensions[0], dtype=jnp.uint64)
        for var_idx in term_indices:
            val = extensions[var_idx]
            product = mod_mul_64(product, val, q)
        total_sum = mod_add_64(total_sum, product, q)
    return total_sum


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
    diff = mod_sub(one_eval, zero_eval, q, bit_width=64)
    prod = mod_mul(diff, target_eval, q, bit_width=64)
    return mod_add(prod, zero_eval, q, bit_width=64)


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



@partial(jax.jit, static_argnums=(3,))
def compute_round_points_kernel_64(tables_tuple, x_points, q, expr_indices):
    """
    JIT-optimized 64-bit kernel using sequential additions for MLE updates.
    """
    exts_0 = [t[0::2] for t in tables_tuple]
    exts_1 = [t[1::2] for t in tables_tuple]
    diffs = [mod_sub_64(o, e, q) for e, o in zip(exts_0, exts_1)]
    
    degree = len(x_points) - 1
    round_poly = []

    # x = 0
    current_exts = exts_0
    comp = compute_composition_indexed_64(expr_indices, current_exts, q)
    sum_val = mod_sum_64(comp, q)
    round_poly.append(sum_val)

    # x = 1
    current_exts = exts_1
    comp = compute_composition_indexed_64(expr_indices, current_exts, q)
    sum_val = mod_sum_64(comp, q)
    round_poly.append(sum_val)

    # x = 2 ... degree
    for i in range(2, degree + 1):
        current_exts = [mod_add_64(cur, d, q) for cur, d in zip(current_exts, diffs)]
        comp = compute_composition_indexed_64(expr_indices, current_exts, q)
        sum_val = mod_sum_64(comp, q)
        round_poly.append(sum_val)

    return jnp.stack(round_poly)


@jax.jit
def fold_tables_kernel_64(tables_tuple, challenge, q):
    """Vectorized folding for 64-bit tables."""
    new_tables = []
    for table in tables_tuple:
        evens = table[0::2]
        odds = table[1::2]
        new_tables.append(mle_update_64(evens, odds, challenge, q=q))
    return tuple(new_tables)


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """
    GPU-optimized 64-bit sumcheck using JAX and sequential evaluation.
    """
    degree = max(len(term) for term in expression)
    q = jnp.asarray(q, dtype=jnp.uint64)
    challenges = jnp.asarray(challenges, dtype=jnp.uint64)
    x_points = jnp.arange(degree + 1, dtype=jnp.uint64)
    
    # 1. Pre-index the expression
    standard_names = ("a", "b", "c", "d", "e", "g")
    active_names = [n for n in standard_names if n in eval_tables]
    for n in eval_tables:
        if n not in active_names:
            active_names.append(n)
            
    name_to_idx = {name: i for i, name in enumerate(active_names)}
    expr_indices = tuple(tuple(name_to_idx[var] for var in term) for term in expression)
    
    # 2. Prepare initial tables
    current_tables = tuple(jnp.asarray(eval_tables[name], dtype=jnp.uint64) for name in active_names)
    
    all_round_evals = []

    for r in range(num_rounds):
        # 3. Compute all points for this round's univariate polynomial
        round_poly = compute_round_points_kernel_64(current_tables, x_points, q, expr_indices)
        all_round_evals.append(round_poly)

        # 4. Fold tables for the next round
        if r < num_rounds - 1:
            current_tables = fold_tables_kernel_64(current_tables, challenges[r], q)

    # Round 0 evaluations give the initial sum: g_1(0) + g_1(1)
    claim0 = mod_add_64(all_round_evals[0][0], all_round_evals[0][1], q)

    return jnp.asarray(claim0, dtype=jnp.uint64), jnp.stack(all_round_evals)


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