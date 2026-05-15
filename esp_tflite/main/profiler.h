#pragma once

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {

// Maximum number of events that this class can keep track of. The
// CustomProfiler will abort if AddEvent is called more than kMaxEvents number
// of times. Increase this number if you need more events.
template <int kMaxEvents = 256, int kMaxUniqueTags = 20>
class CustomProfiler : public MicroProfilerInterface {
 public:
  CustomProfiler() = default;
  virtual ~CustomProfiler() = default;

  uint32_t BeginEvent(const char* tag) {
    if (num_events_ == kMaxEvents) {
      MicroPrintf(
          "CustomProfiler errored out because total number of events exceeded the "
          "maximum of %d.",
          kMaxEvents);
      TFLITE_ASSERT_FALSE;
    }

    events_[num_events_].tag = tag;
    events_[num_events_].start_ticks = GetCurrentTimeTicks();
    events_[num_events_].end_ticks = events_[num_events_].start_ticks - 1;
    return num_events_++;
  }

  void EndEvent(uint32_t event_handle) {
    TFLITE_DCHECK(event_handle < kMaxEvents);
    events_[event_handle].end_ticks = GetCurrentTimeTicks();

    for (int i = 0; i < num_unique_tags_; i++) {
      if (strcmp(total_ticks_per_tag_[i].tag, events_[event_handle].tag) == 0) {
        total_ticks_per_tag_[i].ticks +=
            events_[event_handle].end_ticks - events_[event_handle].start_ticks;
        return;
      }
    }

    if (num_unique_tags_ < kMaxUniqueTags) {
      total_ticks_per_tag_[num_unique_tags_].tag = events_[event_handle].tag;
      total_ticks_per_tag_[num_unique_tags_].ticks =
          events_[event_handle].end_ticks - events_[event_handle].start_ticks;
      num_unique_tags_++;
    } else {
      MicroPrintf(
          "CustomProfiler errored out because total number of unique tags "
          "exceeded the maximum of %d.",
          kMaxUniqueTags);
      TFLITE_ASSERT_FALSE;
    }
  }

  void LogPrecise() const {
  #if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    for (int i = 0; i < num_events_; ++i) {
      uint32_t ticks = events_[i].end_ticks - events_[i].start_ticks;
      MicroPrintf("%s took %u us.", events_[i].tag, ticks);
    }
  #endif
  }

  void LogGrouped() const {
  #if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    for (int i = 0; i < num_unique_tags_; ++i) {
      uint32_t ticks = total_ticks_per_tag_[i].ticks;
      MicroPrintf("%s took %u us.", total_ticks_per_tag_[i].tag, ticks);
    }
  #endif
  }

  void ClearEvents() {
    num_events_ = 0;
    num_unique_tags_ = 0;
  }


 private:
  struct Event {
    const char* tag;
    uint32_t start_ticks;
    uint32_t end_ticks;
  };
  Event events_[kMaxEvents];
  int num_events_ = 0;

  struct TicksPerTag {
    const char* tag;
    uint32_t ticks;
  };
  TicksPerTag total_ticks_per_tag_[kMaxUniqueTags] = {};
  int num_unique_tags_ = 0;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite
