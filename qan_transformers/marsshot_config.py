from dataclasses import dataclass

@dataclass
class MarsshotConfig:
    # Phase 1: Foundation Layer
    attention_mode: str = 'octonionic'         # 'octonionic' | 'projected' | 'standard'
    temperature_mode: str = 'tropical'         # 'tropical' | 'fixed' | 'learned'
    topology_loss_weight: float = 0.1          # persistent homology loss coefficient
    
    # Phase 2: Core Upgrades
    firewall_mode: str = 'motivic'              # 'motivic' | 'cech' | 'disabled'
    spectral_pages: int = 3                     # max spectral sequence pages
    cache_compression: str = 'rg_flow'          # 'rg_flow' | 'morse' | 'none'
    compression_level: float = 0.1              # KV cache compression ratio dial
    
    # Phase 3: Architecture Layer
    use_derived_composition: bool = True
    use_conformal_attention: bool = True
    symplectic_steps: int = 4                   # leapfrog integration steps
    
    # Phase 4: Integration Layer
    adapter_type: str = 'galois'                # 'galois' | 'cayley' | 'lora'
    galois_group: str = 'Z8'                    # 'Z4' | 'Z8' | 'S3' | 'S4'
    use_braiding: bool = True
    braiding_curriculum_steps: int = 1000       # steps before full q/t annealing
