#pragma once

namespace ray_tracing
{
// Hitable objects.
uint32_t constexpr kHitableObjectSphereType = 0;

// Materials.
uint32_t constexpr kMaterialMatteType = 0;
uint32_t constexpr kMaterialMetalType = 1;
uint32_t constexpr kMaterialGlassType = 2;

// Light sources.
uint32_t constexpr kLightSourceDirectionalType = 0;
}  // namespace ray_tracing
