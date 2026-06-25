"""Tests for Floquet wormhole oscillation schedulers."""

import mlx.core as mx
import pytest

from ultrametric_ce.floquet import AdaptiveFloquetScheduler, FloquetScheduler


class TestFloquetScheduler:
    """Tests for the basic FloquetScheduler."""

    def test_get_phase_returns_in_range(self):
        """Phase values should always be in [0, 1] (cos^2)."""
        sched = FloquetScheduler(omega=1.0)
        for _ in range(100):
            phase = sched.get_phase()
            assert 0.0 <= phase <= 1.0, f"Phase {phase} out of [0, 1]"

    def test_phase_oscillates(self):
        """Phase should not be constant — it must change over steps."""
        sched = FloquetScheduler(omega=1.0)
        phases = [sched.get_phase() for _ in range(20)]
        unique_phases = set(round(p, 6) for p in phases)
        assert len(unique_phases) > 1, "Phase is constant, expected oscillation"

    def test_reset_resets_step_counter(self):
        """After reset(), phase sequence should replay from the start."""
        sched = FloquetScheduler(omega=2.0)
        first_run = [sched.get_phase() for _ in range(5)]
        sched.reset()
        second_run = [sched.get_phase() for _ in range(5)]
        for a, b in zip(first_run, second_run):
            assert abs(a - b) < 1e-6, f"Phase mismatch after reset: {a} vs {b}"

    def test_omega_zero_gives_constant_phase(self):
        """With omega=0 the phase is cos^2(0) = 1.0 for every step."""
        sched = FloquetScheduler(omega=0.0)
        for _ in range(10):
            assert abs(sched.get_phase() - 1.0) < 1e-6


class TestAdaptiveFloquetScheduler:
    """Tests for the CFI-adaptive Floquet scheduler."""

    def test_get_phase_returns_mx_array(self):
        """Phase should be returned as an mx.array."""
        sched = AdaptiveFloquetScheduler(omega_init=1.0)
        phase = sched.get_phase()
        assert isinstance(phase, mx.array)

    def test_phase_in_range(self):
        """Phase values should be in [0, 1]."""
        sched = AdaptiveFloquetScheduler(omega_init=1.0)
        for _ in range(50):
            phase = sched.get_phase()
            val = float(phase)
            assert 0.0 <= val <= 1.0 + 1e-6, f"Phase {val} out of [0, 1]"

    def test_high_cfi_increases_omega(self):
        """When CFI is persistently high (>1.0), omega should increase."""
        sched = AdaptiveFloquetScheduler(omega_init=1.0, adaptation_rate=0.1)
        initial_omega = float(sched.omega)
        # Feed high CFI values many times to cross the EMA threshold
        for _ in range(50):
            sched.get_phase(cfi=5.0)
        final_omega = float(sched.omega)
        assert final_omega > initial_omega, (
            f"Expected omega to increase with high CFI: {initial_omega} -> {final_omega}"
        )

    def test_low_cfi_decreases_omega(self):
        """When CFI is persistently low (<0.3), omega should decrease."""
        sched = AdaptiveFloquetScheduler(omega_init=5.0, adaptation_rate=0.1)
        initial_omega = float(sched.omega)
        # Feed low CFI values many times to cross the EMA threshold
        for _ in range(50):
            sched.get_phase(cfi=0.01)
        final_omega = float(sched.omega)
        assert final_omega < initial_omega, (
            f"Expected omega to decrease with low CFI: {initial_omega} -> {final_omega}"
        )

    def test_omega_clamped(self):
        """Omega should remain within [0.1, 10.0] regardless of CFI."""
        sched = AdaptiveFloquetScheduler(omega_init=1.0, adaptation_rate=0.5)
        for _ in range(200):
            sched.get_phase(cfi=100.0)
        assert float(sched.omega) <= 10.0 + 1e-6

        sched2 = AdaptiveFloquetScheduler(omega_init=1.0, adaptation_rate=0.5)
        for _ in range(200):
            sched2.get_phase(cfi=0.0)
        assert float(sched2.omega) >= 0.1 - 1e-6

    def test_reset_resets_step(self):
        """reset() should bring the step counter back to 0."""
        sched = AdaptiveFloquetScheduler(omega_init=1.0)
        for _ in range(10):
            sched.get_phase()
        sched.reset()
        assert sched._step == 0
