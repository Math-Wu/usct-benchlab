"""First-arrival bent-ray physics using a Cartesian fast-marching method.

Solve |grad T| = s with positive, cell-centered slowness in s/m. The accepted
upwind dependencies form a causal tape. Differentiating this *same discrete
solve* gives the Jacobian; reverse accumulation gives its exact transpose.
The default is first order; optional mixed second-order upwinding retains its
own exact discrete derivative. Point-source singularities and interfaces can
still reduce global convergence order. This is not r-Wave's shooting solver.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators.base import Linearization


def interpolation(shape: tuple[int, int], coordinates: np.ndarray):
    """Bilinear interpolation on cell-center indices, and its scatter weights."""
    points = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    if not np.all(np.isfinite(points)):
        raise ValueError("interpolation coordinates must be finite")
    if np.any(points < 0) or np.any(points > np.array(shape) - 1):
        raise ValueError("interpolation points lie outside the computational grid")
    low = np.minimum(np.floor(points).astype(int), np.array(shape) - 2)
    high_weight = points - low
    indices, weights = [], []
    for iy, ix in ((0, 0), (0, 1), (1, 0), (1, 1)):
        indices.append((low[:, 0] + iy) * shape[1] + low[:, 1] + ix)
        wy = high_weight[:, 0] if iy else 1 - high_weight[:, 0]
        wx = high_weight[:, 1] if ix else 1 - high_weight[:, 1]
        weights.append(wy * wx)
    return np.array(indices).T, np.array(weights).T


@dataclass(frozen=True)
class MarchingTape:
    times: np.ndarray
    order: np.ndarray
    parents: np.ndarray
    weights: np.ndarray
    local_derivative: np.ndarray

    def tangent(self, perturbation: np.ndarray) -> np.ndarray:
        from usctbench.operators._marching import tangent_accumulate

        ds = np.asarray(perturbation, dtype=float).ravel()
        return tangent_accumulate(
            self.order, self.parents, self.weights, self.local_derivative, ds
        )


def fast_march(
    slowness: np.ndarray,
    spacing: tuple[float, float],
    source: np.ndarray,
    *,
    compiled=True,
    spatial_order=1,
):
    """Causal first- or mixed second-order marching with off-grid source seeding.

    Four surrounding nodes receive local constant-slowness source values. Their
    analytic derivatives are retained. Source seeding and the grid error converge
    under refinement; no straight-ray projector is used anywhere in this solve.
    """
    if (
        isinstance(spatial_order, (bool, np.bool_))
        or not isinstance(spatial_order, (int, np.integer))
        or spatial_order not in (1, 2)
    ):
        raise ValueError("spatial_order must be 1 or 2")
    model = np.asarray(slowness, dtype=float)
    if model.ndim != 2 or min(model.shape) < 2:
        raise ValueError("fast marching requires a 2-D grid of at least 2 by 2")
    if not np.all(np.isfinite(model)) or np.any(model <= 0):
        raise ValueError("slowness must be finite and positive")
    h = np.asarray(spacing, dtype=float)
    if h.shape != (2,) or np.any(h <= 0) or not np.all(np.isfinite(h)):
        raise ValueError("spacing must be finite and positive")
    ny, nx = model.shape
    flat = model.ravel()
    size = flat.size
    times = np.full(size, np.inf)
    accepted = np.zeros(size, dtype=bool)
    parents = np.full((size, 2), -1, dtype=int)
    weights = np.zeros((size, 2))
    local = np.zeros(size)
    seeds, _ = interpolation(model.shape, np.asarray(source).reshape(1, 2))
    if compiled or spatial_order == 2:
        from usctbench.operators._marching import marching_arrays

        march = marching_arrays
        if not compiled:
            march = getattr(march, "py_func", march)
        arrays = march(
            model, h, np.asarray(source, dtype=float), np.unique(seeds), spatial_order
        )
        return MarchingTape(*arrays)
    order = []
    for node in np.unique(seeds):
        coordinate = np.array(divmod(int(node), nx))
        distance = float(np.linalg.norm((coordinate - source) * h))
        times[node] = distance * flat[node]
        local[node] = distance
        accepted[node] = True
    order.extend(sorted(np.unique(seeds), key=lambda node: times[node]))
    heap = []

    def neighbors(node):
        y, x = divmod(node, nx)
        if y:
            yield node - nx
        if y + 1 < ny:
            yield node + nx
        if x:
            yield node - 1
        if x + 1 < nx:
            yield node + 1

    def update(node):
        if accepted[node]:
            return
        y, x = divmod(node, nx)
        candidates = []
        for axis, pair in enumerate(
            (
                (node - nx if y else -1, node + nx if y + 1 < ny else -1),
                (node - 1 if x else -1, node + 1 if x + 1 < nx else -1),
            )
        ):
            available = [p for p in pair if p >= 0 and accepted[p]]
            if available:
                parent = min(available, key=lambda p: times[p])
                candidates.append((times[parent], parent, h[axis]))
        if not candidates:
            return
        candidates.sort()
        a, parent, step = candidates[0]
        value = a + step * flat[node]
        chosen = [candidates[0]]
        if len(candidates) == 2 and value > candidates[1][0]:
            b, _, step_b = candidates[1]
            wa, wb = 1 / step**2, 1 / step_b**2
            # Shift by a before solving the quadratic to avoid cancellation.
            diff = b - a
            disc = (wa + wb) * flat[node] ** 2 - wa * wb * diff**2
            value = a + (wb * diff + np.sqrt(max(0.0, disc))) / (wa + wb)
            chosen = candidates
        if value >= times[node]:
            return
        times[node] = value
        parents[node] = -1
        weights[node] = 0
        denominator = sum((value - t) / step**2 for t, _, step in chosen)
        local[node] = flat[node] / denominator
        for index, (t, p, step) in enumerate(chosen):
            parents[node, index] = p
            weights[node, index] = (value - t) / step**2 / denominator
        heapq.heappush(heap, (value, node))

    for seed in order:
        for node in neighbors(int(seed)):
            update(node)
    while heap:
        value, node = heapq.heappop(heap)
        if accepted[node] or value != times[node]:
            continue
        accepted[node] = True
        order.append(node)
        for neighbor in neighbors(node):
            update(neighbor)
    if len(order) != size:
        raise RuntimeError("fast marching did not reach all computational nodes")
    return MarchingTape(times, np.asarray(order), parents, weights, local)


class EikonalJacobian:
    """Frozen-active-set derivative with respect to slowness, not sound speed."""

    def __init__(self, model, tapes):
        self.model = model
        self.grid = model.grid
        self.tapes = tuple(tapes)
        self.n_rays = model.n_rays
        self.ray_shape = model.ray_shape

    def forward(self, image: np.ndarray) -> np.ndarray:
        perturbation = self.model.extend(image, background=0.0)
        result = []
        for tape in self.tapes:
            field = tape.tangent(perturbation)
            result.append(self.model.sample_receivers(field))
        return np.asarray(result).ravel()

    def adjoint(self, data: np.ndarray) -> np.ndarray:
        return eikonal_adjoint(self, data)


class EikonalForward:
    """Nonlinear first-arrival forward map for an arbitrary 2-D acquisition.

    Geometry uses [y, x] meters and GridSpec's *cell-edge* origin. Transducers
    outside the image are included by padding with fixed background water.
    With calibrate=True, homogeneous discretization error is removed using an
    identical numerical water solve. The derivative is unchanged by calibration.
    """

    def __init__(
        self,
        grid: GridSpec,
        geometry: GeometrySpec,
        *,
        background_speed_mps: float = 1500.0,
        calibrate: bool = True,
        spatial_order: int = 1,
    ):
        if not np.isfinite(background_speed_mps) or background_speed_mps <= 0:
            raise ValueError("background_speed_mps must be finite and positive")
        if (
            isinstance(spatial_order, (bool, np.bool_))
            or not isinstance(spatial_order, (int, np.integer))
            or spatial_order not in (1, 2)
        ):
            raise ValueError("spatial_order must be 1 or 2")
        self.spatial_order = spatial_order
        positions = np.vstack((geometry.tx_pos_m, geometry.rx_pos_m))
        if not np.all(np.isfinite(positions)):
            raise ValueError("transducer positions must be finite")
        self.grid = grid
        self.geometry = geometry
        self.background_slowness = 1 / background_speed_mps
        self.calibrate = bool(calibrate)
        h = np.array(grid.spacing_m)
        first_center = np.array(grid.origin_m) + h / 2
        coordinates = (positions - first_center) / h
        before = np.maximum(0, 2 - np.floor(coordinates.min(axis=0)).astype(int))
        after = np.maximum(
            0, np.ceil(coordinates.max(axis=0)).astype(int) + 3 - grid.shape
        )
        self.padding = tuple((int(a), int(b)) for a, b in zip(before, after))
        self.shape = tuple(np.array(grid.shape) + before + after)
        self.crop = tuple(slice(int(a), int(a + n)) for a, n in zip(before, grid.shape))
        self.source_coordinates = coordinates[: len(geometry.tx_pos_m)] + before
        self.receiver_indices, self.receiver_weights = interpolation(
            self.shape, coordinates[len(geometry.tx_pos_m) :] + before
        )
        self.ray_shape = (len(geometry.tx_pos_m), len(geometry.rx_pos_m))
        self.n_rays = int(np.prod(self.ray_shape))
        self._reference = None
        self.forward_calls = 0

    def extend(self, image: np.ndarray, *, background: float) -> np.ndarray:
        image = np.asarray(image, dtype=float)
        if image.shape != self.grid.shape:
            raise ValueError("image shape must match the reconstruction grid")
        if not np.all(np.isfinite(image)):
            raise ValueError("image contains non-finite entries")
        return np.pad(image, self.padding, constant_values=background)

    def sample_receivers(self, field: np.ndarray) -> np.ndarray:
        return np.sum(
            np.asarray(field).ravel()[self.receiver_indices] * self.receiver_weights,
            axis=1,
        )

    def fields(self, slowness: np.ndarray) -> tuple[MarchingTape, ...]:
        extended = self.extend(slowness, background=self.background_slowness)
        return tuple(
            fast_march(
                extended, self.grid.spacing_m, p, spatial_order=self.spatial_order
            )
            for p in self.source_coordinates
        )

    def _raw(self, slowness):
        tapes = self.fields(slowness)
        values = np.asarray([self.sample_receivers(t.times) for t in tapes]).ravel()
        return values, tapes

    def linearize(self, slowness: np.ndarray) -> Linearization:
        self.forward_calls += 1
        values, tapes = self._raw(slowness)
        if self.calibrate:
            if self._reference is None:
                water = np.full(self.grid.shape, self.background_slowness)
                raw, _ = self._raw(water)
                distance = np.linalg.norm(
                    self.geometry.tx_pos_m[:, None] - self.geometry.rx_pos_m[None, :],
                    axis=-1,
                ).ravel()
                self._reference = raw - distance * self.background_slowness
            values = values - self._reference
        return Linearization(values, EikonalJacobian(self, tapes))

    def forward(self, slowness: np.ndarray) -> np.ndarray:
        return self.linearize(slowness).value


def eikonal_adjoint(jacobian, data: np.ndarray) -> np.ndarray:
    from usctbench.operators._marching import reverse_accumulate

    model = jacobian.model
    values = np.asarray(data, dtype=float)
    if values.size != model.n_rays:
        raise ValueError("data size must match (n_tx, n_rx)")
    if not np.all(np.isfinite(values)):
        raise ValueError("adjoint data must be finite")
    values = values.reshape(model.ray_shape)
    result = np.zeros(model.shape)
    for tape, receiver_values in zip(jacobian.tapes, values):
        sensitivity = np.zeros(tape.times.size)
        np.add.at(
            sensitivity,
            model.receiver_indices.ravel(),
            (receiver_values[:, None] * model.receiver_weights).ravel(),
        )
        gradient = reverse_accumulate(
            tape.order, tape.parents, tape.weights, tape.local_derivative, sensitivity
        )
        result += gradient.reshape(model.shape)
    return result[model.crop].copy()
