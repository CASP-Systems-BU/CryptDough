#pragma once

#include "aggregation.h"
#include "common.h"
#include "core/containers/a_shared_vector.h"
#include "core/containers/b_shared_vector.h"

namespace cdough::operators {

/**
 * @brief Computes tumbling window identifiers for stream processing.
 *
 * A tumbling window divides the stream into non-overlapping windows of fixed size.
 * This function computes the window identifier for each element by dividing the
 * key (typically a timestamp or sequence number) by the window size.
 *
 * @tparam Share The underlying data type of the shared vectors.
 * @tparam EVector Share container type.
 * @tparam Engine Secure computation engine type.
 *
 * @param key Input vector containing keys/timestamps for window assignment.
 * @param window_size The size of each tumbling window.
 * @param res Output vector that will contain the computed window identifiers.
 */
template <typename Share, typename EVector, typename Engine>
static void tumbling_window(ASharedVector<Share, EVector, Engine>& key, const Share& window_size,
                            ASharedVector<Share, EVector, Engine>& res) {
    res = key / window_size;
}

/**
 * @brief Marks the start of new gap-based sessions in a timestamp sequence.
 *
 * This function identifies session boundaries based on gaps in timestamps.
 * A new session starts when the gap between consecutive timestamps exceeds
 * the specified threshold. The first element is always marked as a session start.
 *
 * @tparam Share The underlying data type of the shared vectors.
 * @tparam EVector Share container type.
 * @tparam Engine Secure computation engine type.
 *
 * @param timestamp Input vector of timestamps in ascending order.
 * @param session_start Output boolean vector marking session start positions.
 * @param gap The maximum allowed gap between timestamps within a session.
 */
template <typename Share, typename EVector, typename Engine>
void mark_gap_session(ASharedVector<Share, EVector, Engine>& timestamp,
                      BSharedVector<Share, EVector, Engine>& session_start, const Share& gap) {
    auto& engine = timestamp.engine;

    // Set the first index to 1
    Vector<Share> one({1});
    session_start.slice(0, 1) = engine.public_share_b(one);

    auto shared_gap_vec = engine.public_share_a(Vector<Share>(1, gap));
    auto shared_gap_vec_extended = shared_gap_vec.repeated_subset_reference(timestamp.size() - 1);

    ASharedVector<Share, EVector, Engine> pair_wise_gap =
        timestamp.slice(0, timestamp.size() - 1) + shared_gap_vec_extended - timestamp.slice(1);

    auto pair_wise_gap_b = pair_wise_gap.a2b();
    session_start.slice(1) = pair_wise_gap_b->ltz();
}

/**
 * @brief Creates gap-based session windows for stream aggregation.
 *
 * This function implements session windowing based on timestamp gaps. Sessions
 * are identified using the mark_gap_session function, then window identifiers
 * are computed using reverse aggregation to assign the same window ID to all
 * elements within a session.
 *
 * @tparam Share The underlying data type of the shared vectors.
 * @tparam EVector Share container type.
 * @tparam Engine Secure computation engine type.
 *
 * @param keys Vector of key vectors used for aggregation operations.
 * @param timestamp_a Arithmetic shared vector of timestamps.
 * @param timestamp_b Boolean shared vector of timestamps (for multiplexing).
 * @param window_id Output vector that will contain the computed window identifiers.
 * @param gap The maximum allowed gap between timestamps within a session.
 */
template <typename Share, typename EVector, typename Engine>
void gap_session_window(std::vector<BSharedVector<Share, EVector, Engine>>& keys,
                        ASharedVector<Share, EVector, Engine>& timestamp_a,
                        BSharedVector<Share, EVector, Engine>& timestamp_b,
                        BSharedVector<Share, EVector, Engine>& window_id, const Share& gap) {
    auto& engine = timestamp_a.engine;

    mark_gap_session(timestamp_a, window_id, gap);

    auto shared_one = engine.public_share_b(Vector<Share>({-1}));

    window_id =
        multiplex(window_id, shared_one.repeated_subset_reference(window_id.size()), timestamp_b);

    cdough::aggregators::aggregate(keys, {{window_id, window_id, cdough::aggregators::max}}, {},
                                cdough::aggregators::Direction::Forward);
}

/**
 * @brief Marks the start of new threshold-based sessions.
 *
 * This function identifies session boundaries based on a threshold applied to
 * a function result (e.g., sensor readings, activity levels). A new session
 * starts when the function result exceeds the threshold and the previous
 * element was below the threshold.
 *
 * @tparam Share The underlying data type of the shared vectors.
 * @tparam EVector Share container type.
 * @tparam Engine Secure computation engine type.
 *
 * @param function_res Input vector containing function results to compare against threshold.
 * @param session_start Output boolean vector marking session start positions.
 * @param potential_window Output boolean vector indicating elements above threshold.
 * @param threshold The threshold value for session detection.
 */
template <typename Share, typename EVector, typename Engine>
void mark_threshold_session(BSharedVector<Share, EVector, Engine>& function_res,
                            BSharedVector<Share, EVector, Engine>& session_start,
                            BSharedVector<Share, EVector, Engine>& potential_window,
                            const Share& threshold) {
    auto& engine = function_res.engine;

    Vector<Share> threshold_vec({threshold});
    auto shared_threshold_vec = engine.public_share_b(threshold_vec);

    potential_window =
        function_res > shared_threshold_vec.repeated_subset_reference(function_res.size());

    session_start.slice(1) =
        potential_window.slice(1) &&
        (potential_window.slice(1).xor_1(potential_window.slice(0, potential_window.size() - 1)));
}

/**
 * @brief Creates threshold-based session windows for stream aggregation.
 *
 * This function implements session windowing based on threshold detection.
 * Sessions are identified when function results exceed a threshold, and
 * window identifiers are computed using aggregation with the potential_window
 * as a selection mask to only include elements that meet the threshold criteria.
 *
 * @tparam Share The underlying data type of the shared vectors.
 * @tparam EVector Share container type.
 * @tparam Engine Secure computation engine type.
 *
 * @param keys Vector of key vectors used for aggregation operations.
 * @param function_res Input vector containing function results for threshold comparison.
 * @param timestamp_b Boolean shared vector of timestamps (for multiplexing).
 * @param window_id Output vector that will contain the computed window identifiers.
 * @param gap The threshold value for session detection (note: parameter name is misleading).
 */
template <typename Share, typename EVector, typename Engine>
void threshold_session_window(std::vector<BSharedVector<Share, EVector, Engine>>& keys,
                              BSharedVector<Share, EVector, Engine>& function_res,
                              BSharedVector<Share, EVector, Engine>& timestamp_b,
                              BSharedVector<Share, EVector, Engine>& window_id, const Share& gap) {
    auto& engine = function_res.engine;
    BSharedVector<Share, EVector, Engine> potential_window(window_id.size(), engine);

    mark_threshold_session(function_res, window_id, potential_window, gap);

    Vector<Share> neg_one({-1});
    BSharedVector<Share, EVector, Engine> shared__one = engine.public_share_b(neg_one);
    BSharedVector<Share, EVector, Engine> extended_shared__one =
        shared__one.repeated_subset_reference(window_id.size());

    window_id = multiplex(window_id, extended_shared__one, timestamp_b);

    cdough::aggregators::aggregate(keys, {{window_id, window_id, cdough::aggregators::max}}, {},
                                potential_window);
}
}  // namespace cdough::operators
