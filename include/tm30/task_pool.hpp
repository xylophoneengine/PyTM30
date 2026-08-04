#pragma once

/// @file task_pool.hpp
/// Persistent worker pool for repeated parallel batch evaluations
/// (Phase 2 of the opt-in n_workers feature).
///
/// Phase 1 spawns threads per call; that is a small, one-time tax for a
/// single one-off batch, but pure waste for a long-lived TM30Calc making
/// many repeated eval() calls. This pool amortizes thread creation over
/// the object's lifetime: workers are created eagerly at construction
/// and process a sequence of unrelated run_chunked() jobs over time.

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tm30 {

/// Persistent worker pool (move-only).
///
/// Each run_chunked() call partitions [0, n) into min(workers, n)
/// balanced contiguous chunks - the SAME partition math as the
/// spawn-per-call path in tm30.cpp - and blocks until every chunk has
/// finished. Workers persist across calls.
///
/// Single-caller model: exactly one thread may be inside run_chunked()
/// at a time (the Python binding guarantees this - eval() is called
/// under the GIL with one caller). Because run_chunked() is blocking,
/// workers are always idle between calls, which makes destruction
/// trivially safe: the destructor sets a shutdown flag, wakes every
/// worker, and joins. A worker that is mid-chunk when shutdown fires
/// (only reachable via a caller bug, never via the single-caller model)
/// finishes its bounded chunk before exiting, so joins always terminate.
///
/// Synchronization: a monotonic job id (epoch) distinguishes jobs.
/// Workers wait until the current job id differs from the one they last
/// processed, so a finished worker can never re-execute a chunk of a
/// job whose caller-side fn may already be out of scope, and a job that
/// is published while a worker is between chunks is never lost (the
/// predicate is re-checked under the lock, not just on the notify).
///
/// Exception safety (fault isolation): a task exception never kills a
/// worker. The first exception of a job is captured and rethrown on the
/// calling thread after all chunks complete, so a pool survives a
/// failing call at full strength for the next call.
class TaskPool {
public:
  /// Create `n_workers` persistent worker threads eagerly.
  explicit TaskPool(std::size_t n_workers);

  /// Signal shutdown, wake and join all workers. Never hangs: workers
  /// are idle between calls (blocking run_chunked), and a mid-chunk
  /// worker finishes its bounded chunk before exiting.
  ~TaskPool();

  TaskPool(const TaskPool &) = delete;
  TaskPool &operator=(const TaskPool &) = delete;

  /// Execute fn(i) for every i in [0, n), partitioned into contiguous
  /// chunks across the workers. Blocks until all chunks are done.
  /// Rethrows the first task exception (if any) after all chunks
  /// finish; the pool remains usable for subsequent calls.
  void run_chunked(std::size_t n, const std::function<void(std::size_t)> &fn);

  /// Number of worker threads.
  std::size_t size() const noexcept { return workers_.size(); }

private:
  void worker_loop(std::size_t worker_id);

  /// Per-job state, all guarded by mu_. Only run_chunked() writes it;
  /// workers read it after observing a new job id.
  struct JobState {
    std::size_t id = 0;         // monotonic job epoch
    std::size_t n = 0;          // iteration count
    std::size_t chunk_base = 0; // partition params (see tm30.cpp)
    std::size_t chunk_extra = 0;
    const std::function<void(std::size_t)> *fn = nullptr;
    std::size_t completed = 0;      // chunks finished this job
    std::exception_ptr first_error; // first task exception, if any
  };

  mutable std::mutex mu_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  bool shutdown_ = false;
  JobState job_;

  std::vector<std::thread> workers_;
};

} // namespace tm30
