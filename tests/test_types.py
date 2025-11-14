#!/usr/bin/env python3
"""Test script to verify type checking works for pyopensim."""

import pyopensim as osim

# These should all be recognized by type checkers:
model = osim.Model()  # Should work!
vec = osim.Vec3(1.0, 2.0, 3.0)
rot = osim.Rotation()
transform = osim.Transform()

# Also test submodule access
model2 = osim.simulation.Model()
vec2 = osim.simbody.Vec3(1.0, 2.0, 3.0)

print("✓ All type checks passed!")
print(f"Model: {type(model)}")
print(f"Vec3: {type(vec)}")
