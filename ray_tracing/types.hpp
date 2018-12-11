#pragma once

namespace ray_tracing
{
// Hitable objects.
uint8_t constexpr kHitableObjectSphereType = 0;

// Materials.
uint8_t constexpr kMaterialMatteType = 0;
uint8_t constexpr kMaterialMetalType = 1;
uint8_t constexpr kMaterialGlassType = 2;

// Light sources.
uint8_t constexpr kLightSourceDirectionalType = 0;
}  // namespace ray_tracing
