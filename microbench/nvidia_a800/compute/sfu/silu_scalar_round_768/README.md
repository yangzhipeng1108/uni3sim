# A800 768-thread scalar-SiLU round probe

This independent mechanism probe runs a register-only recurrent scalar SiLU chain for 64 rounds per thread, with 768 threads and a 1 KiB static-SMEM residency class. It contains no production tensor traffic or complete operator. A frozen capture may publish only one stable SM-clock-cycle-per-CTA-round mechanism measurement and one effective one-versus-two-CTA overlap capacity; it must not become a per-grid LUT, additive stage time, duration fit, or direct Forward input.
