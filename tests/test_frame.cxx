#include <array>
#include <dx2/image/frame.hpp>
#include <gtest/gtest.h>
#include <numeric>
#include <variant>

using dx2::ConstFrameView;
using dx2::Frame;
using dx2::FrameShape;
using dx2::FrameView;
using dx2::PixelType;

// A Frame allocates exactly one frame of storage and describes it correctly.
TEST(FrameTest, FrameAllocatesStorageForItsTypeAndShape) {
  Frame frame(PixelType::UInt16, FrameShape{4, 6});
  EXPECT_EQ(frame.size_bytes(), 4u * 6u * 2u)
      << "Storage size differs from slow * fast * pixel_size";
  FrameView view = frame.view();
  EXPECT_EQ(view.type(), PixelType::UInt16) << "View lost the pixel type";
  EXPECT_EQ(view.shape(), (FrameShape{4, 6})) << "View lost the shape";
  EXPECT_EQ(view.num_pixels(), 24u) << "num_pixels is not slow * fast";
}

// A default-constructed Frame is empty but still yields a consistent view.
TEST(FrameTest, DefaultFrameIsEmptyAndViewable) {
  Frame frame;
  EXPECT_EQ(frame.size_bytes(), 0u) << "Default frame owns storage";
  EXPECT_EQ(frame.view().num_pixels(), 0u) << "Default view has pixels";
}

// Writing through the flat span and reading through the mdspan must agree
// on row-major layout: element (row, col) is at index row * fast + col.
TEST(FrameTest, FlatSpanAndMdspanShareRowMajorLayout) {
  Frame frame(PixelType::UInt32, FrameShape{3, 5});
  std::span<uint32_t> flat = frame.view().as<uint32_t>();
  ASSERT_EQ(flat.size(), 15u) << "Flat span has the wrong element count";
  std::iota(flat.begin(), flat.end(), 100u);

  auto md = frame.view().as_mdspan<uint32_t>();
  EXPECT_EQ(md.extent(0), 3u) << "mdspan slow extent is wrong";
  EXPECT_EQ(md.extent(1), 5u) << "mdspan fast extent is wrong";
  EXPECT_EQ(md(2, 3), 100u + 2 * 5 + 3)
      << "mdspan (2, 3) is not the row-major element";
}

// Asking for the wrong element type is a programming error and must throw
// with both type names in the message.
TEST(FrameTest, AsWithTheWrongTypeThrows) {
  Frame frame(PixelType::UInt16, FrameShape{2, 2});
  try {
    frame.view().as<uint32_t>();
    FAIL() << "as<uint32_t>() on a uint16 frame did not throw";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("uint16"), std::string::npos)
        << "Message does not name the frame's type: " << msg;
    EXPECT_NE(msg.find("uint32"), std::string::npos)
        << "Message does not name the requested type: " << msg;
  }
}

// A view is one frame exactly; a buffer of any other size is rejected at
// construction so no later access can run off the end.
TEST(FrameTest, ViewOverWrongSizedBufferThrows) {
  std::array<std::byte, 10> buffer{};
  EXPECT_THROW((FrameView{buffer, PixelType::UInt16, FrameShape{2, 2}}),
               std::invalid_argument)
      << "A 10-byte buffer was accepted for 8 bytes of pixels";
  EXPECT_NO_THROW((FrameView{std::span(buffer).first(8), PixelType::UInt16,
                             FrameShape{2, 2}}))
      << "An exactly sized buffer was rejected";
}

// A caller-supplied buffer that is not aligned for the pixel type cannot be
// reinterpreted, so typed access must refuse it.
TEST(FrameTest, AsOnAMisalignedBufferThrows) {
  alignas(8) std::array<std::byte, 9> buffer{};
  FrameView view{std::span(buffer).subspan(1, 8), PixelType::UInt32,
                 FrameShape{1, 2}};
  EXPECT_THROW(view.as<uint32_t>(), std::runtime_error)
      << "as<uint32_t>() accepted a buffer offset by one byte";
}

// A writable view converts to a read-only one, and typed access through the
// read-only view yields const elements.
TEST(FrameTest, WritableViewConvertsToConstView) {
  Frame frame(PixelType::Int16, FrameShape{2, 3});
  frame.view().as<int16_t>()[5] = -7;
  ConstFrameView cview = frame.view();
  std::span<const int16_t> pixels = cview.as<int16_t>();
  EXPECT_EQ(pixels[5], -7) << "Const view does not see the written pixel";
  static_assert(
      std::is_same_v<decltype(cview.as_mdspan<int16_t>())::element_type,
                     const int16_t>,
      "const view must yield const elements");
}

// visit dispatches to a typed mdspan matching the run-time PixelType, so a
// generic algorithm is written once.
TEST(FrameTest, VisitDispatchesOnThePixelType) {
  Frame frame(PixelType::Float32, FrameShape{2, 2});
  auto flat = frame.view().as<float>();
  flat[0] = 1.5f;
  flat[1] = 2.5f;
  flat[2] = 3.0f;
  flat[3] = 4.0f;

  double sum = dx2::visit(frame.view(), [](auto md) {
    double total = 0.0;
    for (size_t i = 0; i < md.extent(0); ++i) {
      for (size_t j = 0; j < md.extent(1); ++j) {
        total += md(i, j);
      }
    }
    return total;
  });
  EXPECT_DOUBLE_EQ(sum, 11.0) << "visit summed the wrong elements";
}

// as_variant holds the alternative for the frame's type, for callers using
// std::visit directly.
TEST(FrameTest, AsVariantHoldsTheMatchingAlternative) {
  Frame frame(PixelType::UInt16, FrameShape{1, 1});
  dx2::FrameVariant v = dx2::as_variant(frame.view());
  EXPECT_TRUE(std::holds_alternative<dx2::mdspan_2d<uint16_t>>(v))
      << "Variant does not hold the uint16 mdspan";
  dx2::ConstFrameVariant cv = dx2::as_variant(ConstFrameView(frame.view()));
  EXPECT_TRUE(std::holds_alternative<dx2::mdspan_2d<const uint16_t>>(cv))
      << "Const variant does not hold the const uint16 mdspan";
}
