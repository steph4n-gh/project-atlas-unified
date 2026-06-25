"""Floquet-modulated wormhole oscillation for ultrametric diffusion.

Provides periodic wormhole gate modulation so that wormhole connections
between distant tree branches open and close with a learnable frequency,
implementing natural temporal attention decay.
"""
import mlx.core as mx
import mlx.nn as nn

class FloquetScheduler:
    """Manages periodic wormhole oscillation phase."""
    def __init__(self, omega: float = 1.0):
        self.omega = omega
        self._step = 0
    
    def get_phase(self) -> float:
        """Returns cos^2(omega * t / 2) modulation factor."""
        phase = float(mx.cos(mx.array(self.omega * self._step / 2.0)) ** 2)
        self._step += 1
        return phase
    
    def reset(self):
        self._step = 0

class AdaptiveFloquetScheduler(nn.Module):
    """Floquet scheduler with learnable frequency that adapts based on CFI feedback.
    
    When CFI is high (attention fracturing), oscillation speeds up to
    cycle through wormhole configurations faster, searching for coherent topologies.
    When CFI is low (stable attention), oscillation slows down to maintain
    the current effective graph topology.
    """
    def __init__(self, omega_init: float = 1.0, adaptation_rate: float = 0.1):
        super().__init__()
        self.omega = mx.array([omega_init])
        self.adaptation_rate = adaptation_rate
        self._step = 0
        self._cfi_ema = 0.0
    
    def get_phase(self, cfi: float = None) -> mx.array:
        """Returns differentiable cos^2(omega * t / 2) modulation factor.
        
        Args:
            cfi: Optional current Cohomology Fracture Index. If provided,
                 adapts omega based on CFI trend.
        Returns:
            phase: mx.array scalar in [0, 1]
        """
        if cfi is not None:
            # EMA smoothing of CFI signal
            self._cfi_ema = 0.9 * self._cfi_ema + 0.1 * cfi
            # High CFI -> increase omega (faster oscillation to search for stability)
            # Low CFI -> decrease omega (slow down, we found a good topology)
            if self._cfi_ema > 1.0:
                self.omega = self.omega * (1.0 + self.adaptation_rate)
            elif self._cfi_ema < 0.3:
                self.omega = self.omega * (1.0 - self.adaptation_rate * 0.5)
            # Clamp omega to reasonable range
            self.omega = mx.clip(self.omega, 0.1, 10.0)
        
        phase = mx.cos(self.omega * self._step / 2.0) ** 2
        self._step += 1
        return phase
    
    def reset(self):
        self._step = 0
