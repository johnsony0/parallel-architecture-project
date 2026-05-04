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


def get_mont_params_32(q):
    """Compute Montgomery parameters for 32-bit modulus."""
    q = jnp.uint32(q)
    # Newton's method for modular inverse modulo 2^32
    q_prime = jnp.uint32(1)
    for _ in range(5):
        q_prime = q_prime * (jnp.uint32(2) - q * q_prime)
    q_prime = -q_prime
    
    # R = 2^32. R_mod_q = 2^32 % q = (2^32 - q) % q = -q % q
    R_mod_q = jnp.uint32((jnp.uint64(1) << 32) % jnp.uint64(q))
    R2_mod_q = jnp.uint32((jnp.uint64(R_mod_q) * jnp.uint64(R_mod_q)) % jnp.uint64(q))
    return q_prime, R_mod_q, R2_mod_q


def redc_32(T, q, q_prime):
    """Montgomery reduction: returns T * R^-1 mod q."""
    m = jnp.uint32(T) * q_prime
    t = (T >> 32) + ((jnp.uint64(jnp.uint32(T)) + jnp.uint64(m) * jnp.uint64(q)) >> 32)
    return jnp.where(t >= q, jnp.uint32(t - q), jnp.uint32(t))


def mont_mul_32(a_bar, b_bar, q, q_prime):
    """Montgomery multiplication: returns (a_bar * b_bar * R^-1) mod q."""
    T = jnp.uint64(a_bar) * jnp.uint64(b_bar)
    return redc_32(T, q, q_prime)


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

def mle_update_32_mont(zero_eval_bar, one_eval_bar, target_eval_bar, q, q_prime):
    """MLE update in Montgomery form."""
    diff_bar = mod_sub_32(one_eval_bar, zero_eval_bar, q)
    prod_bar = mont_mul_32(diff_bar, target_eval_bar, q, q_prime)
    return mod_add_32(prod_bar, zero_eval_bar, q)


def compute_composition_indexed_mont(expr_indices, extensions_bar, q, q_prime, R_mod_q):
    """Computes composition in Montgomery form."""
    total_sum_bar = jnp.zeros_like(extensions_bar[0], dtype=jnp.uint32)
    for term_indices in expr_indices:
        # Start with 1 in Montgomery form, which is R mod q
        product_bar = jnp.full_like(extensions_bar[0], R_mod_q, dtype=jnp.uint32)
        for var_idx in term_indices:
            val_bar = extensions_bar[var_idx]
            product_bar = mont_mul_32(product_bar, val_bar, q, q_prime)
        total_sum_bar = mod_add_32(total_sum_bar, product_bar, q)
    return total_sum_bar

@partial(jax.jit, static_argnums=(4,))
def compute_round_points_kernel_mont(tables_tuple_bar, x_points_bar, q, q_prime, expr_indices, R_mod_q):
    """JIT-optimized kernel using Montgomery reduction."""
    def eval_at_x(x_bar):
        extensions_bar = []
        for table_bar in tables_tuple_bar:
            evens_bar = table_bar[0::2]
            odds_bar = table_bar[1::2]
            extensions_bar.append(mle_update_32_mont(evens_bar, odds_bar, x_bar, q, q_prime))
        
        comp_bar = compute_composition_indexed_mont(expr_indices, extensions_bar, q, q_prime, R_mod_q)
        # Sum is still in Montgomery form
        sum_bar = (jnp.sum(comp_bar.astype(jnp.uint64)) % jnp.uint64(q)).astype(jnp.uint32)
        # Convert back from Montgomery form: redc(sum_bar) = sum_bar * R^-1 mod q
        return redc_32(jnp.uint64(sum_bar), q, q_prime)

    return jax.vmap(eval_at_x)(x_points_bar)


@jax.jit
def fold_tables_kernel_mont(tables_tuple_bar, challenge_bar, q, q_prime):
    """Vectorized folding in Montgomery form."""
    new_tables_bar = []
    for table_bar in tables_tuple_bar:
        evens_bar = table_bar[0::2]
        odds_bar = table_bar[1::2]
        new_tables_bar.append(mle_update_32_mont(evens_bar, odds_bar, challenge_bar, q, q_prime))
    return tuple(new_tables_bar)


def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    GPU-optimized 32-bit sumcheck using Montgomery reduction.
    """
    degree = max(len(term) for term in expression)
    q = jnp.asarray(q, dtype=jnp.uint32)
    challenges = jnp.asarray(challenges, dtype=jnp.uint32)
    x_points = jnp.arange(degree + 1, dtype=jnp.uint32)
    
    # Montgomery setup
    q_prime, R_mod_q, R2_mod_q = get_mont_params_32(q)
    
    # Helper to convert to Montgomery form: (x * R) mod q = redc(x * R^2)
    def to_mont(x):
        return mont_mul_32(x, R2_mod_q, q, q_prime)

    # 1. Pre-index the expression for JIT compatibility
    standard_names = ("a", "b", "c", "d", "e", "g")
    active_names = [n for n in standard_names if n in eval_tables]
    for n in eval_tables:
        if n not in active_names:
            active_names.append(n)
            
    name_to_idx = {name: i for i, name in enumerate(active_names)}
    expr_indices = tuple(tuple(name_to_idx[var] for var in term) for term in expression)
    
    # 2. Prepare initial tables in Montgomery form
    current_tables_bar = tuple(to_mont(jnp.asarray(eval_tables[name], dtype=jnp.uint32)) for name in active_names)
    challenges_bar = to_mont(challenges)
    x_points_bar = to_mont(x_points)
    
    all_round_evals = []

    for r in range(num_rounds):
        # 3. Compute all points for this round's univariate polynomial
        round_poly = compute_round_points_kernel_mont(
            current_tables_bar, x_points_bar, q, q_prime, expr_indices, R_mod_q
        )
        all_round_evals.append(round_poly)

        # 4. Fold tables for the next round
        if r < num_rounds - 1:
            current_tables_bar = fold_tables_kernel_mont(current_tables_bar, challenges_bar[r], q, q_prime)

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