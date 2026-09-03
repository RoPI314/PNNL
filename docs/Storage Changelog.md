# Save Format Changelog

## v3 (Sep 1st, 2026)
- Added activations `ACTIVATION_SIGMOID` and `ACTIVATION_TANH`.
- This doesn't break v2's format, but these activations were not supported prior.

## v2 (Sep 1st, 2026)
- `activation` field shrunk from `int` (4 bytes) to `uint8_t` (1 byte) per layer.
- Breaking: v1 files are unreadable under v2 (field size mismatch shifts every subsequent byte).

## v1 (Sep 1st, 2026)
- Initial format.
- Layout: `version (uint16_t)`, `num_layers (int)`, then per layer:
  `input_size (int)`, `output_size (int)`, `activation (int)`, `weights (float[])`, `biases (float[])`.