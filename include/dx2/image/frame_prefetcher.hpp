/**
 * @file frame_prefetcher.hpp
 * @brief Worker threads that read frames ahead of a consumer.
 *
 * Workers claim frames of the range in index order, wait for each to
 * exist, read it and post it; the consumer takes finished frames with
 * next(), in order or as they complete. Any ImageSource that is safe to
 * read from several threads can sit underneath.
 *
 * At most a window of frames is claimed and not yet taken, so the window
 * times the frame size bounds memory; it must be at least the worker
 * count. Buffers are pooled because the kernel zero-fills a fresh
 * multi-megabyte allocation page by page on first write, which costs
 * about as much as the decode; recycle() returns an Item's buffer for
 * reuse, and a dropped Item is freed.
 *
 * A frame that misses frame_timeout, or an error from the source, ends
 * the run: next() throws it once, then reports the end. stop() ends the
 * run early and the destructor joins the workers. An Item owns its bytes.
 */

#pragma once

#include "dx2/image/frame.hpp"
#include "dx2/image/image_source.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace dx2 {

class FramePrefetcher {
public:
  /// The order next() hands frames out in.
  enum class Delivery : uint8_t { InOrder, AnyOrder };
  /// What each Item carries: decoded pixels, or the stored chunk.
  enum class Payload : uint8_t { Decoded, Raw };

  struct Options {
    size_t workers = 4;
    /// Frames claimed and not yet taken by the consumer; at least workers.
    size_t window = 8;
    Delivery delivery = Delivery::InOrder;
    Payload payload = Payload::Decoded;
    /// Longest wait for one frame to be written before the run fails.
    std::chrono::milliseconds frame_timeout{60000};
    /// Interval between checks for a frame that is not there yet.
    std::chrono::milliseconds poll{20};
  };

  /// One frame as read. For Payload::Decoded `frame` holds the pixels.
  /// For Payload::Raw it holds the stored chunk as a {1, stored_bytes}
  /// run of bytes, with header_bytes of them before the compressed
  /// payload.
  struct Item {
    size_t index;
    ChunkInfo info;
    Frame frame;
    size_t header_bytes = 0;
  };

  /**
   * Starts reading frames [first, last) of source.
   * @throws std::invalid_argument if the window is smaller than the
   * workers, there are no workers, or raw payload is asked of a source
   * that offers no raw chunks.
   */
  FramePrefetcher(ImageSource &source, size_t first, size_t last,
                  Options options);
  FramePrefetcher(ImageSource &source, size_t first, size_t last);
  ~FramePrefetcher();
  FramePrefetcher(const FramePrefetcher &) = delete;
  FramePrefetcher &operator=(const FramePrefetcher &) = delete;

  /// Blocks for the next frame. nullopt once every frame of the range has
  /// been handed out or stop() was called; a failed read is thrown once.
  std::optional<Item> next();
  /// Ends the run; workers stop at their next check and next() reports
  /// the end once the frames already finished are gone.
  void stop() noexcept;
  /// Returns an Item's buffer to the pool. Only Items from this
  /// prefetcher may be given back, as the pool assumes its own size.
  void recycle(Item &&item);

  const Options &options() const { return options_; }

private:
  void work(std::stop_token stop);
  Item read(size_t index);
  std::unique_ptr<std::byte[]> take_buffer();

  ImageSource &source_;
  size_t first_;
  size_t last_;
  Options options_;

  std::mutex mutex_;
  std::condition_variable claim_cv_;
  std::condition_variable ready_cv_;
  size_t next_claim_;
  size_t delivered_ = 0;
  std::map<size_t, Item> ready_;
  std::exception_ptr error_;
  bool stopped_ = false;
  /// Bytes of every pooled buffer: a frame, or the largest raw chunk.
  size_t buffer_bytes_;
  std::vector<std::unique_ptr<std::byte[]>> pool_;
  std::vector<std::jthread> workers_;
};

inline FramePrefetcher::FramePrefetcher(ImageSource &source, size_t first,
                                        size_t last)
    : FramePrefetcher(source, first, last, Options{}) {}

} // namespace dx2
