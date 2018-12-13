#pragma once

#include <glm/vec3.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace demo
{
class Palette
{
public:
  static std::vector<uint32_t> const & Blue()
  {
    static std::vector<uint32_t> kColors = {0x011f4b, 0x03396c, 0x005b96, 0x6497b1, 0xb3cde0,
                                            0x2a4d69, 0x4b86b4, 0xadcbe3, 0xe7eff6, 0x63ace5,
                                            0x6e7f80, 0x536872, 0x708090, 0x536878, 0x36454f,
                                            0x3385c6, 0x4279a3, 0x476c8a, 0x49657b, 0x7f8e9e};
    return kColors;
  }

  static std::vector<uint32_t> const & Warm()
  {
    static std::vector<uint32_t> kColors = {0xeee3e7, 0xead5dc, 0xeec9d2, 0xf4b6c2, 0xf6abb6,
                                            0xf6cd61, 0xfe8a71, 0xfe9c8f, 0xfeb2a8, 0xfec8c1,
                                            0xfad9c1, 0xf9caa7, 0xee4035, 0xf37736, 0xfdf498,
                                            0xff77aa, 0xff99cc, 0xffbbee, 0xff5588, 0xff3377};
    return kColors;
  }

  static std::vector<uint32_t> const & Green()
  {
    static std::vector<uint32_t> kColors = {0x0e9aa7, 0x3da4ab, 0x009688, 0x35a79c, 0x54b2a9,
                                            0x65c3ba, 0x83d0c9, 0x7bc043, 0x96ceb4, 0x88d8b0,
                                            0xa8e6cf, 0xdcedc1, 0x00b159, 0xE5FCC2, 0x9DE0AD,
                                            0x45ADA8, 0xC8C8A9, 0x83AF9B, 0xA8E6CE, 0xDCEDC2};
    return kColors;
  }

  static std::vector<uint32_t> const & Yellow()
  {
    static std::vector<uint32_t> kColors = {0xfed766, 0xf6cd61, 0xfdf498, 0xffeead, 0xffcc5c,
                                            0xffa700, 0xffc425, 0xedc951, 0xf1c27d, 0xffdbac};
    return kColors;
  }

  static std::vector<uint32_t> const & Red()
  {
    static std::vector<uint32_t> kColors = {0xfe4a49, 0x851e3e, 0xee4035, 0xff6f69, 0xd62d20,
                                            0xff8b94, 0xd11141, 0xcc2a36, 0xEC2049, 0xFE4365};
    return kColors;
  }

  template<typename Generator>
  static glm::vec3 Random(Generator & generator, std::vector<uint32_t> const & palette)
  {
    std::uniform_int_distribution<> rnd(0, static_cast<int>(palette.size() - 1));
    return Convert(palette[rnd(generator)]);
  }

  template<typename Generator>
  static glm::vec3 RandomFromAll(Generator & generator)
  {
    static std::vector<std::vector<uint32_t>> kAll = {Blue(), Warm(), Green(), Yellow(), Red()};

    std::uniform_int_distribution<> rnd(0, static_cast<int>(kAll.size() - 1));
    return Random(generator, kAll[rnd(generator)]);
  }

private:
  static glm::vec3 Convert(uint32_t color)
  {
    return glm::vec3(static_cast<uint8_t>(color >> 16 & 0xFF) / 255.0f,
                     static_cast<uint8_t>(color >> 8 & 0xFF) / 255.0f,
                     static_cast<uint8_t>(color & 0xFF) / 255.0f);
  }
};
}  // namespace demo
