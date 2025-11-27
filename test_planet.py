#!/usr/bin/env python3
"""
Test script to verify PlaNet integration in CarDreamer
"""

import sys
import jax
import jax.numpy as jnp

# Add dreamerv3 to path
sys.path.insert(0, './dreamerv3')

from dreamerv3 import nets, ninjax as nj


def test_planet_basic():
    """Test basic PlaNet functionality"""
    print("=" * 60)
    print("Testing PlaNet Basic Functionality")
    print("=" * 60)

    # Initialize PlaNet
    batch_size = 4
    deter_dim = 128
    action_dim = 2
    embed_dim = 64
    seq_len = 10

    config = {
        'deter': deter_dim,
        'stoch': 32,
        'classes': 32,
        'unroll': False,
        'initial': 'learned',
        'action_clip': 1.0,
        'units': 64,
        'act': 'silu',
        'norm': 'layer',
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create PlaNet instance
    with nj.init_scope():
        planet = nets.PlaNet(**config, name='planet')

    print("\n✓ PlaNet instance created successfully")

    # Test initial state
    print("\nTesting initial state generation...")
    state = planet.initial(batch_size)
    print(f"  State keys: {state.keys()}")
    print(f"  Deter shape: {state['deter'].shape}")
    print(f"  Stoch shape: {state['stoch'].shape}")
    assert state['deter'].shape == (batch_size, deter_dim)
    print("✓ Initial state generation works")

    # Test img_step (imagination step)
    print("\nTesting imagination step (img_step)...")
    action = jnp.zeros((batch_size, action_dim))
    next_state = planet.img_step(state, action)
    print(f"  Next state keys: {next_state.keys()}")
    print(f"  Next deter shape: {next_state['deter'].shape}")
    assert next_state['deter'].shape == (batch_size, deter_dim)
    print("✓ Imagination step works")

    # Test obs_step (observation update)
    print("\nTesting observation step (obs_step)...")
    embed = jnp.zeros((batch_size, embed_dim))
    is_first = jnp.zeros(batch_size)
    post, prior = planet.obs_step(state, action, embed, is_first)
    print(f"  Post state keys: {post.keys()}")
    print(f"  Prior state keys: {prior.keys()}")
    print(f"  Post deter shape: {post['deter'].shape}")
    assert post['deter'].shape == (batch_size, deter_dim)
    assert prior['deter'].shape == (batch_size, deter_dim)
    print("✓ Observation step works")

    # Test observe (full sequence)
    print("\nTesting full observation sequence...")
    embed_seq = jnp.zeros((batch_size, seq_len, embed_dim))
    action_seq = jnp.zeros((batch_size, seq_len, action_dim))
    is_first_seq = jnp.zeros((batch_size, seq_len))
    post_seq, prior_seq = planet.observe(embed_seq, action_seq, is_first_seq)
    print(f"  Post sequence deter shape: {post_seq['deter'].shape}")
    print(f"  Prior sequence deter shape: {prior_seq['deter'].shape}")
    assert post_seq['deter'].shape == (batch_size, seq_len, deter_dim)
    print("✓ Full observation sequence works")

    # Test imagine (rollout)
    print("\nTesting imagination rollout...")
    action_rollout = jnp.zeros((seq_len, batch_size, action_dim))
    prior_rollout = planet.imagine(action_rollout, state)
    print(f"  Prior rollout deter shape: {prior_rollout['deter'].shape}")
    assert prior_rollout['deter'].shape == (batch_size, seq_len, deter_dim)
    print("✓ Imagination rollout works")

    # Test loss functions
    print("\nTesting loss functions...")
    dyn_loss = planet.dyn_loss(post, prior, impl='mse')
    print(f"  Dynamic loss shape: {dyn_loss.shape}")
    print(f"  Dynamic loss mean: {dyn_loss.mean():.4f}")
    assert dyn_loss.shape == (batch_size,)

    rep_loss = planet.rep_loss(post, prior, impl='none')
    print(f"  Representation loss shape: {rep_loss.shape}")
    print(f"  Representation loss mean: {rep_loss.mean():.4f}")
    assert rep_loss.shape == (batch_size,)
    assert jnp.allclose(rep_loss, 0.0), "Rep loss should be zero for PlaNet"
    print("✓ Loss functions work")

    print("\n" + "=" * 60)
    print("✅ All PlaNet tests passed!")
    print("=" * 60)


def test_planet_vs_rssm():
    """Compare PlaNet and RSSM interfaces"""
    print("\n" + "=" * 60)
    print("Testing PlaNet vs RSSM Interface Compatibility")
    print("=" * 60)

    batch_size = 4
    deter_dim = 128
    action_dim = 2
    embed_dim = 64

    config = {
        'deter': deter_dim,
        'stoch': 32,
        'classes': 32,
        'unroll': False,
        'initial': 'learned',
        'action_clip': 1.0,
        'units': 64,
        'act': 'silu',
        'norm': 'layer',
    }

    print("\nCreating both models...")
    with nj.init_scope():
        planet = nets.PlaNet(**config, name='planet')
        rssm = nets.RSSM(**config, name='rssm')

    print("✓ Both models created")

    # Test that they have the same interface
    print("\nTesting interface compatibility...")

    # Initial state
    planet_state = planet.initial(batch_size)
    rssm_state = rssm.initial(batch_size)

    assert set(planet_state.keys()) == set(rssm_state.keys()), \
        f"State keys differ: {planet_state.keys()} vs {rssm_state.keys()}"
    print("  ✓ initial() returns same keys")

    # Check methods exist
    methods = ['initial', 'observe', 'imagine', 'img_step', 'obs_step',
               'get_dist', 'dyn_loss', 'rep_loss']

    for method in methods:
        assert hasattr(planet, method), f"PlaNet missing method: {method}"
        assert hasattr(rssm, method), f"RSSM missing method: {method}"

    print(f"  ✓ All {len(methods)} required methods present")

    print("\n" + "=" * 60)
    print("✅ Interface compatibility verified!")
    print("=" * 60)


def test_planet_memory():
    """Test memory efficiency of PlaNet vs RSSM"""
    print("\n" + "=" * 60)
    print("Testing Memory Footprint")
    print("=" * 60)

    # This is a rough estimate - actual memory depends on JAX compilation
    batch_size = 16
    deter_dim = 1024

    config = {
        'deter': deter_dim,
        'stoch': 32,
        'classes': 32,
        'unroll': False,
        'initial': 'learned',
        'action_clip': 1.0,
        'units': 256,
        'act': 'silu',
        'norm': 'layer',
    }

    print("\nCreating models...")
    with nj.init_scope():
        planet = nets.PlaNet(**config, name='planet')
        rssm = nets.RSSM(**config, name='rssm')

    # Get state sizes
    planet_state = planet.initial(batch_size)
    rssm_state = rssm.initial(batch_size)

    planet_size = sum(s.size * s.dtype.itemsize for s in jax.tree_util.tree_leaves(planet_state))
    rssm_size = sum(s.size * s.dtype.itemsize for s in jax.tree_util.tree_leaves(rssm_state))

    print(f"\nState memory footprint (batch_size={batch_size}):")
    print(f"  PlaNet: {planet_size / 1024:.2f} KB")
    print(f"  RSSM:   {rssm_size / 1024:.2f} KB")

    if planet_size < rssm_size:
        reduction = (1 - planet_size / rssm_size) * 100
        print(f"  ✓ PlaNet uses {reduction:.1f}% less memory")
    else:
        print(f"  Note: Sizes are similar for state representation")

    print("=" * 60)


if __name__ == '__main__':
    print("\n🚀 Starting PlaNet Integration Tests\n")

    try:
        # Set random seed
        import numpy as np
        np.random.seed(42)

        # Run tests
        test_planet_basic()
        test_planet_vs_rssm()
        test_planet_memory()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! PlaNet is ready to use!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Try training with: bash train_dm3.sh 2000 0 --configs planet --task carla_four_lane")
        print("2. Read PLANET_INTEGRATION.md for detailed usage instructions")
        print("3. Compare performance between RSSM and PlaNet on your tasks")
        print()

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
