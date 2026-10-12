#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// Row-major [candidate, pick]. Times must already include station statics.
// Preserve the reference: median origin correction, then minimum RMS;
// strict comparison retains the first candidate in an exact tie.
extern "C" EXPORT int score_candidates(const double* times, const double* obs,
    std::int64_t nc, std::int64_t np, std::int64_t* best,
    double* shifts, double* scores) noexcept {
    if (!times || !obs || !best || !shifts || !scores || nc < 1 || np < 1)
        return 1;
    try {
        for (std::int64_t p = 0; p < np; ++p)
            if (!std::isfinite(obs[p])) return 2;
        std::vector<double> residual(np);
        *best = -1;
        double minimum = std::numeric_limits<double>::infinity();
        for (std::int64_t c = 0; c < nc; ++c) {
            shifts[c] = std::numeric_limits<double>::quiet_NaN();
            scores[c] = std::numeric_limits<double>::infinity();
            bool valid = true;
            for (std::int64_t p = 0; p < np; ++p) {
                const double t = times[c * np + p];
                if (!std::isfinite(t)) { valid = false; break; }
                residual[p] = obs[p] - t;
                if (!std::isfinite(residual[p])) { valid = false; break; }
            }
            if (!valid) continue;
            std::sort(residual.begin(), residual.end());
            double shift = residual[np / 2];
            if (np % 2 == 0) shift = (residual[np / 2 - 1] + shift) * 0.5;
            if (!std::isfinite(shift)) continue;
            double sum = 0;
            for (std::int64_t p = 0; p < np; ++p) {
                const double r = obs[p] - (times[c * np + p] + shift);
                sum += r * r;
            }
            const double score = std::sqrt(sum / np);
            if (!std::isfinite(score)) continue;
            shifts[c] = shift;
            scores[c] = score;
            if (score < minimum) { minimum = score; *best = c; }
        }
        return 0;
    } catch (...) { return 3; }
}

// Reference downhill 26-neighbour backtrace. Returns step count, -1 if
// no decreasing route reaches the receiver, -2 on invalid arguments.
// Each segment is assigned to the departing node, as in the reference.
extern "C" EXPORT std::int64_t trace_ray(const double* field,
    std::int64_t nz, std::int64_t ny, std::int64_t nx,
    std::int64_t source, std::int64_t receiver, double spacing,
    std::int64_t capacity, std::int64_t* indices, double* lengths) noexcept {
    const auto total = nz * ny * nx;
    if (!field || !indices || !lengths || nz < 1 || ny < 1 || nx < 1 ||
        source < 0 || source >= total || receiver < 0 || receiver >= total ||
        capacity < 1 || spacing <= 0 || !std::isfinite(spacing)) return -2;
    auto current = source;
    for (std::int64_t step = 0; step < capacity; ++step) {
        if (current == receiver) return step;
        const auto k = current / (ny * nx);
        const auto j = (current / nx) % ny;
        const auto i = current % nx;
        auto next = current;
        double minimum = field[current], distance = 0;
        if (!std::isfinite(minimum)) return -1;
        for (int dk = -1; dk <= 1; ++dk)
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    if (dk == 0 && dj == 0 && di == 0) continue;
                    const auto kk = k + dk, jj = j + dj, ii = i + di;
                    if (kk < 0 || kk >= nz || jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
                    const auto candidate = (kk * ny + jj) * nx + ii;
                    if (field[candidate] < minimum) {
                        minimum = field[candidate]; next = candidate;
                        distance = spacing * std::sqrt(double(dk*dk + dj*dj + di*di));
                    }
                }
        if (next == current) return -1;
        indices[step] = current;
        lengths[step] = distance;
        current = next;
    }
    return current == receiver ? capacity : -1;
}
