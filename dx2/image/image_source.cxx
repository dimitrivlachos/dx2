#include "dx2/image/image_source.hpp"
#include <algorithm>
#include <thread>

namespace dx2 {

bool ImageSource::wait_for_frame(size_t frame,
                                 std::chrono::milliseconds timeout,
                                 std::stop_token stop,
                                 std::chrono::milliseconds poll) {
  using clock = std::chrono::steady_clock;
  const auto deadline = clock::now() + timeout;
  while (true) {
    if (chunk_info(frame)) {
      return true;
    }
    const auto now = clock::now();
    if (stop.stop_requested() || now >= deadline) {
      return false;
    }
    // Sleep for one poll interval or whatever remains of the deadline.
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    std::this_thread::sleep_for(std::min(poll, remaining));
  }
}

} // namespace dx2
