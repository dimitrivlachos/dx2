#include "dx2/image/frame_prefetcher.hpp"
#include <fmt/format.h>
#include <stdexcept>

namespace dx2 {

FramePrefetcher::FramePrefetcher(ImageSource &source, size_t first, size_t last,
                                 Options options)
    : source_(source), first_(first), last_(last), options_(options),
      next_claim_(first) {
  if (options_.workers == 0) {
    throw std::invalid_argument("a prefetcher needs at least one worker");
  }
  if (options_.window < options_.workers) {
    throw std::invalid_argument(
        fmt::format("a window of {} cannot hold a frame for each of {} workers",
                    options_.window, options_.workers));
  }
  if (options_.payload == Payload::Raw && !source_.supports_raw_chunks()) {
    throw std::invalid_argument("the source offers no raw chunks");
  }
  buffer_bytes_ = options_.payload == Payload::Raw
                      ? source_.max_raw_chunk_bytes()
                      : source_.frame_bytes();
  pool_.reserve(options_.window);
  workers_.reserve(options_.workers);
  for (size_t i = 0; i < options_.workers; ++i) {
    workers_.emplace_back([this](std::stop_token stop) { work(stop); });
  }
}

FramePrefetcher::~FramePrefetcher() {
  stop();
  workers_.clear();
}

void FramePrefetcher::stop() noexcept {
  {
    std::lock_guard lock(mutex_);
    stopped_ = true;
  }
  for (std::jthread &worker : workers_) {
    worker.request_stop();
  }
  claim_cv_.notify_all();
  ready_cv_.notify_all();
}

void FramePrefetcher::work(std::stop_token stop) {
  while (true) {
    // Claim the next frame once the window has room for it.
    size_t index = 0;
    {
      std::unique_lock lock(mutex_);
      claim_cv_.wait(lock, [&] {
        return stopped_ || error_ || next_claim_ >= last_ ||
               next_claim_ < first_ + delivered_ + options_.window;
      });
      if (stopped_ || error_ || next_claim_ >= last_) {
        return;
      }
      index = next_claim_++;
    }

    // Wait for the frame, read it, and post it; any failure ends the run
    // for every worker.
    try {
      if (!source_.wait_for_frame(index, options_.frame_timeout, stop,
                                  options_.poll)) {
        if (stop.stop_requested()) {
          return;
        }
        throw std::runtime_error(
            fmt::format("frame {} did not arrive within {} ms", index,
                        options_.frame_timeout.count()));
      }
      Item item = read(index);
      {
        std::lock_guard lock(mutex_);
        ready_.emplace(index, std::move(item));
      }
      ready_cv_.notify_all();
    } catch (...) {
      {
        std::lock_guard lock(mutex_);
        error_ = std::current_exception();
      }
      ready_cv_.notify_all();
      claim_cv_.notify_all();
      return;
    }
  }
}

std::unique_ptr<std::byte[]> FramePrefetcher::take_buffer() {
  {
    std::lock_guard lock(mutex_);
    if (!pool_.empty()) {
      std::unique_ptr<std::byte[]> buffer = std::move(pool_.back());
      pool_.pop_back();
      return buffer;
    }
  }
  return std::make_unique_for_overwrite<std::byte[]>(buffer_bytes_);
}

void FramePrefetcher::recycle(Item &&item) {
  std::unique_ptr<std::byte[]> buffer = item.frame.release();
  if (!buffer) {
    return;
  }
  std::lock_guard lock(mutex_);
  if (pool_.size() < options_.window) {
    pool_.push_back(std::move(buffer));
  }
}

FramePrefetcher::Item FramePrefetcher::read(size_t index) {
  Item item;
  item.index = index;
  std::unique_ptr<std::byte[]> buffer = take_buffer();
  if (options_.payload == Payload::Raw) {
    // The chunk lands at the start of a buffer sized for the largest; the
    // frame then spans just the stored bytes.
    const RawChunk raw =
        source_.read_raw_chunk(index, {buffer.get(), buffer_bytes_});
    item.frame =
        Frame(std::move(buffer), PixelType::UInt8, {1, raw.bytes.size()});
    item.info = raw.info;
    item.header_bytes = raw.header_bytes;
    return item;
  }
  item.frame = Frame(std::move(buffer), source_.pixel_type(), source_.shape());
  source_.read_frame_into(index, item.frame.view());
  item.info = source_.chunk_info(index).value_or(ChunkInfo{index});
  return item;
}

std::optional<FramePrefetcher::Item> FramePrefetcher::next() {
  std::unique_lock lock(mutex_);
  const auto deliverable = [&] {
    return options_.delivery == Delivery::AnyOrder
               ? !ready_.empty()
               : ready_.contains(first_ + delivered_);
  };
  const auto finished = [&] { return first_ + delivered_ >= last_; };
  ready_cv_.wait(
      lock, [&] { return error_ || stopped_ || deliverable() || finished(); });

  // An error is thrown once; afterwards the run reads as stopped.
  if (error_) {
    const std::exception_ptr error = std::exchange(error_, nullptr);
    stopped_ = true;
    lock.unlock();
    claim_cv_.notify_all();
    std::rethrow_exception(error);
  }
  if (!deliverable()) {
    return std::nullopt;
  }
  auto found = options_.delivery == Delivery::AnyOrder
                   ? ready_.begin()
                   : ready_.find(first_ + delivered_);
  Item item = std::move(found->second);
  ready_.erase(found);
  ++delivered_;
  lock.unlock();
  // Taking a frame frees a window slot for the workers.
  claim_cv_.notify_one();
  return item;
}

} // namespace dx2
