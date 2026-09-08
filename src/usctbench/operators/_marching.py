"""Optional compiled loops for the same causal Godunov discretization.

No fastmath, reduced precision or different PDE scheme is used. The uncompiled
functions remain executable for exact regression comparisons.
"""

import heapq
import numpy as np

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        return lambda function: function


@njit(cache=True)
def update_node(
    node, ny, nx, hy, hx, flat, accepted, times, parents, weights, local, heap
):
    if accepted[node]:
        return
    y, x = node // nx, node % nx
    p0 = -1
    if y > 0 and accepted[node - nx]:
        p0 = node - nx
    if y + 1 < ny and accepted[node + nx] and (p0 < 0 or times[node + nx] < times[p0]):
        p0 = node + nx
    p1 = -1
    if x > 0 and accepted[node - 1]:
        p1 = node - 1
    if x + 1 < nx and accepted[node + 1] and (p1 < 0 or times[node + 1] < times[p1]):
        p1 = node + 1
    h0, h1 = hy, hx
    if p0 < 0:
        p0, p1, h0, h1 = p1, p0, h1, h0
    elif p1 >= 0 and (times[p1] < times[p0] or (times[p1] == times[p0] and p1 < p0)):
        p0, p1, h0, h1 = p1, p0, h1, h0
    if p0 < 0:
        return
    a = times[p0]
    value = a + h0 * flat[node]
    both = p1 >= 0 and value > times[p1]
    if both:
        wa, wb = 1 / h0**2, 1 / h1**2
        diff = times[p1] - a
        disc = (wa + wb) * flat[node] ** 2 - wa * wb * diff**2
        value = a + (wb * diff + np.sqrt(max(0.0, disc))) / (wa + wb)
    if value >= times[node]:
        return
    times[node] = value
    parents[node, 0], parents[node, 1] = p0, -1
    weights[node, 1] = 0.0
    denominator = (value - a) / h0**2
    if both:
        denominator += (value - times[p1]) / h1**2
        parents[node, 1] = p1
        weights[node, 1] = (value - times[p1]) / h1**2 / denominator
    local[node] = flat[node] / denominator
    weights[node, 0] = (value - a) / h0**2 / denominator
    heapq.heappush(heap, (value, node))


@njit(cache=True)
def marching_arrays(model, spacing, source, seeds):
    ny, nx = model.shape
    flat = model.ravel()
    times = np.full(flat.size, np.inf)
    accepted = np.zeros(flat.size, np.bool_)
    parents = np.full((flat.size, 2), -1, np.int64)
    weights = np.zeros((flat.size, 2))
    local = np.zeros(flat.size)
    order = np.empty(flat.size, np.int64)
    seed_order = []
    for node in seeds:
        distance = np.sqrt(
            ((node // nx - source[0]) * spacing[0]) ** 2
            + ((node % nx - source[1]) * spacing[1]) ** 2
        )
        times[node] = distance * flat[node]
        local[node] = distance
        accepted[node] = True
        seed_order.append((times[node], node))
    seed_order.sort()
    count = len(seeds)
    heap = [(0.0, -1)]
    heap.pop()
    for i, pair in enumerate(seed_order):
        node = pair[1]
        order[i] = node
        for candidate in (node - nx, node + nx, node - 1, node + 1):
            if (
                candidate < 0
                or candidate >= flat.size
                or (abs(candidate - node) == 1 and candidate // nx != node // nx)
            ):
                continue
            update_node(
                candidate,
                ny,
                nx,
                spacing[0],
                spacing[1],
                flat,
                accepted,
                times,
                parents,
                weights,
                local,
                heap,
            )
    while heap:
        value, node = heapq.heappop(heap)
        if accepted[node] or value != times[node]:
            continue
        accepted[node] = True
        order[count] = node
        count += 1
        for candidate in (node - nx, node + nx, node - 1, node + 1):
            if (
                candidate < 0
                or candidate >= flat.size
                or (abs(candidate - node) == 1 and candidate // nx != node // nx)
            ):
                continue
            update_node(
                candidate,
                ny,
                nx,
                spacing[0],
                spacing[1],
                flat,
                accepted,
                times,
                parents,
                weights,
                local,
                heap,
            )
    if count != flat.size:
        raise RuntimeError("fast marching did not reach all nodes")
    return times, order, parents, weights, local


@njit(cache=True)
def conservative_transport(order, parents, times, shape, spacing, initial):
    """Positive upwind finite-volume solve of div(I grad(T)) = 0, I=A^2.

    Each interior face has one shared flux. Boundary gradients are extrapolated
    from the inward face, allowing outgoing energy instead of a reflecting wall.
    Source seed intensities are Dirichlet data, not transport unknowns.
    """
    ny, nx = shape
    intensity = np.exp(2 * initial)
    for node in order:
        if parents[node, 0] < 0:
            continue
        y, x = node // nx, node % nx
        incoming, outgoing = 0.0, 0.0
        for axis in range(2):
            stride = nx if axis == 0 else 1
            coordinate = y if axis == 0 else x
            size = ny if axis == 0 else nx
            for sign in (-1, 1):
                neighbor = node + sign * stride
                if coordinate + sign < 0 or coordinate + sign >= size:
                    inward = node - sign * stride
                    outgoing += max(times[node] - times[inward], 0) / spacing[axis] ** 2
                else:
                    flux = (times[neighbor] - times[node]) / spacing[axis] ** 2
                    if flux >= 0:
                        outgoing += flux
                    else:
                        incoming -= flux * intensity[neighbor]
        if outgoing <= 0 or incoming <= 0:
            raise FloatingPointError("nonpositive WKB transport flux")
        intensity[node] = incoming / outgoing
    return 0.5 * np.log(intensity)


@njit(cache=True)
def tangent_accumulate(order, parents, weights, local, perturbation):
    result = np.zeros(local.size)
    for node in order:
        value = local[node] * perturbation[node]
        for k in range(2):
            parent = parents[node, k]
            if parent >= 0:
                value += weights[node, k] * result[parent]
        result[node] = value
    return result


@njit(cache=True)
def reverse_accumulate(order, parents, weights, local, values):
    sensitivity = values.copy()
    gradient = np.zeros(local.size)
    for i in range(len(order) - 1, -1, -1):
        node = order[i]
        gradient[node] += sensitivity[node] * local[node]
        for k in range(2):
            parent = parents[node, k]
            if parent >= 0:
                sensitivity[parent] += sensitivity[node] * weights[node, k]
    return gradient
