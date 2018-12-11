#pragma once

#include <cstdint>

namespace ray_tracing
{
float constexpr kPi = static_cast<float>(3.14159265358979323846);

float constexpr kEps = 1e-5f;

// Hitable objects.
uint8_t constexpr kHitableObjectSphereType = 0;

// Materials.
uint8_t constexpr kMaterialMatteType = 0;
uint8_t constexpr kMaterialMetalType = 1;
uint8_t constexpr kMaterialGlassType = 2;

// Light sources.
uint8_t constexpr kLightSourceDirectionalType = 0;
}  // namespace ray_tracing
