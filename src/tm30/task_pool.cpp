// Persistent worker pool implementation (see task_pool.hpp).
#include "tm30/task_pool.hpp"

#include <algorithm> // std::min

#if defined(__linux__)
#include <pthread.h>
#include <sched.h> // cpu_set_t, CPU_SET, pthread_setaffinity_np
#endif

namespace tm30 {

TaskPool::TaskPool(std::size_t n_workers) {
  // Eager pool creation: workers exist for the pool's whole lifetime.
  // n_workers == 0 is degenerate (run_chunked would return without
  // executing fn) - callers must not create such a pool; the binding
  // layer only constructs pools when n_workers > 1.
  workers_.reserve(n_workers);
  for (std::size_t w = 0; w < n_workers; ++w)
    workers_.emplace_back([this, w] { worker_loop(w); });
}

TaskPool::~TaskPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    shutdown_ = true;
  }
  cv_work_.notify_all();
  for (auto &t : workers_)
    t.join();
}

void TaskPool::run_chunked(std::size_t n,
                           const std::function<void(std::size_t)> &fn) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    // Balanced contiguous chunks, same math as the spawn-per-call path
    // in tm30.cpp: thread t handles
    //   [t*base + min(t, extra), (t+1)*base + min(t+1, extra)).
    // With n < workers this degenerates to 1 iteration per chunk for
    // the first n workers (base = 0, extra = n).
    const std::size_t nchunks = std::min(workers_.size(), n);
    ++job_.id;
    job_.n = n;
    job_.chunk_base = (nchunks == 0) ? 0 : n / nchunks;
    job_.chunk_extra = (nchunks == 0) ? 0 : n % nchunks;
    job_.fn = &fn;
    job_.completed = 0;
    job_.first_error = nullptr;
  }
  cv_work_.notify_all();

  std::unique_lock<std::mutex> lock(mu_);
  cv_done_.wait(lock, [&] {
    return job_.completed >= std::min(workers_.size(), job_.n);
  });

  if (job_.first_error)
    std::rethrow_exception(job_.first_error);
}

void TaskPool::worker_loop(std::size_t worker_id) {
#if defined(__linux__)
  // Pin this worker to a fixed CPU. Between jobs a worker is parked; an
  // unpinned thread can be migrated to a core that drops into a deep
  // idle C-state (mobile APUs: C6/C7 exit costs ~1-2 ms), which would
  // destroy the point of persistent workers - each job would pay a
  // multi-ms wake latency on the critical path. Pinning keeps the wake
  // latency in the tens-of-microseconds range and gives cache locality.
  const unsigned int n_cpus = std::thread::hardware_concurrency();
  if (n_cpus > 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(worker_id % n_cpus), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
#endif

  std::size_t last_done = 0; // id of the last job this worker released
  std::unique_lock<std::mutex> lock(mu_);
  for (;;) {
    // Wait for a NEW job (id differs from the one we last processed) or
    // shutdown. Checking the id - not a boolean flag - makes this safe
    // against both re-execution (a finished worker can never re-enter
    // the same job id) and lost wakeups (a job published while the
    // worker is between chunks satisfies the predicate immediately).
    cv_work_.wait(lock, [&] { return shutdown_ || job_.id != last_done; });
    if (shutdown_)
      return;

    const std::size_t my_id = job_.id;
    const std::size_t nchunks = std::min(workers_.size(), job_.n);
    if (worker_id >= nchunks) {
      // No chunk for this job - release and wait for the next one.
      last_done = my_id;
      continue;
    }

    const std::size_t begin =
        worker_id * job_.chunk_base + std::min(worker_id, job_.chunk_extra);
    const std::size_t end = (worker_id + 1) * job_.chunk_base +
                            std::min(worker_id + 1, job_.chunk_extra);
    const auto *fn = job_.fn;

    lock.unlock(); // run the chunk without holding the pool mutex
    try {
      for (std::size_t i = begin; i < end; ++i)
        (*fn)(i);
    } catch (...) {
      lock.lock();
      if (!job_.first_error)
        job_.first_error = std::current_exception();
      lock.unlock();
    }
    lock.lock();

    ++job_.completed;
    cv_done_.notify_one();

    // Done with this job. Loop back to wait for the id to change -
    // never re-enter this chunk (fn belongs to the caller's stack and
    // may be out of scope once run_chunked() returns).
    last_done = my_id;
  }
}

} // namespace tm30
