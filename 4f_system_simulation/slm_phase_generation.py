import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom


TWO_PI = 2.0 * np.pi


def wrap_phase_to_unit(phase_rad: np.ndarray) -> np.ndarray:
    """Wrap phase from radians to [0, 1) for storage."""
    return np.mod(phase_rad, TWO_PI).astype(np.float32) / TWO_PI


def smooth_modulation_to_unit(
    surface: np.ndarray,
    phase_span: float = 0.75,
    softness: float = 1.8,
    center: float = 0.5,
) -> np.ndarray:
    """
    Map a smooth random surface to a continuous SLM modulation in [0, 1].

    This deliberately avoids modulo wrapping.  The previous wrap-to-unit path
    can place neighboring values near 0 and 1 even when the underlying phase is
    continuous, which creates artificial black/white edges in the stored mask.
    """
    span = float(np.clip(phase_span, 1e-3, 1.0))
    soft = max(float(softness), 1e-3)
    modulation = np.tanh(surface / soft)
    phase_unit = center + 0.5 * span * modulation
    return np.clip(phase_unit, 0.0, 1.0).astype(np.float32)


def percentile_normalize_to_unit(
    surface: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    margin: float = 0.03,
) -> np.ndarray:
    """
    Normalize a smooth surface to [0, 1] without histogram equalization.

    A small percentile clip prevents rare outliers from collapsing contrast,
    while the margin avoids large fully black/white plateaus.
    """
    lo, hi = np.percentile(surface, [lower_percentile, upper_percentile])
    phase_unit = (surface - lo) / (hi - lo + 1e-8)
    phase_unit = np.clip(phase_unit, 0.0, 1.0)
    phase_unit = margin + (1.0 - 2.0 * margin) * phase_unit
    return phase_unit.astype(np.float32)


def generate_smoothed_noise_unit(
    size: int = 1024,
    sigma: float = 8.0,
    seed: int | None = None,
    passes: int = 1,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    margin: float = 0.03,
) -> np.ndarray:
    """
    Generate unit phase by directly smoothing uniform random noise.

    This is intentionally simpler than the Fourier/rank-equalized path: random
    noise -> Gaussian blur -> percentile normalization.
    """
    rng = np.random.default_rng(seed)
    phase = rng.random((size, size), dtype=np.float32)
    for _ in range(max(int(passes), 1)):
        phase = gaussian_filter(phase, sigma=sigma, mode="reflect").astype(np.float32)
    return percentile_normalize_to_unit(
        phase,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        margin=margin,
    )


def resize_unit_phase(
    phase: np.ndarray,
    target_size: int = 1024,
    order: int = 3,
) -> np.ndarray:
    """Resize a unit phase map with bicubic interpolation."""
    if phase.shape == (target_size, target_size):
        return phase.astype(np.float32)

    zoom_y = target_size / phase.shape[0]
    zoom_x = target_size / phase.shape[1]
    resized = zoom(phase, (zoom_y, zoom_x), order=order, mode="reflect")
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def generate_lowres_smoothed_noise_unit(
    lowres_size: int = 256,
    target_size: int = 1024,
    sigma: float = 2.5,
    seed: int | None = None,
    passes: int = 1,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    margin: float = 0.03,
) -> np.ndarray:
    """Generate smooth phase at low resolution, then resize to target size."""
    lowres = generate_smoothed_noise_unit(
        size=lowres_size,
        sigma=sigma,
        seed=seed,
        passes=passes,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        margin=margin,
    )
    return resize_unit_phase(lowres, target_size=target_size, order=3)


def normalized_focus_surface(size: int) -> np.ndarray:
    """Return a zero-mean, unit-std quadratic focus trend."""
    _, r2 = radial_grid(size)
    r2 = r2 / max(float(r2.max()), 1.0)
    focus = r2.astype(np.float32)
    focus -= focus.mean()
    focus /= focus.std() + 1e-8
    return focus


def add_smooth_focus_to_noise(
    noise_phase: np.ndarray,
    focus_strength: float = 0.18,
) -> np.ndarray:
    """
    Add a smooth quadratic focusing trend to an already-smoothed noise phase.

    Positive focus_strength makes the edge brighter than the center, matching a
    gentle lens-like radial phase while keeping the map continuous and bounded.
    """
    focus = normalized_focus_surface(noise_phase.shape[0])
    focused = noise_phase + float(focus_strength) * focus
    return percentile_normalize_to_unit(
        focused,
        lower_percentile=0.5,
        upper_percentile=99.5,
        margin=0.03,
    )


def radial_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized radius and squared radius grids."""
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    x = xx - center
    y = yy - center
    r2 = x * x + y * y
    r_max = np.sqrt(2.0) * center
    r = np.sqrt(r2) / max(r_max, 1.0)
    return r.astype(np.float32), r2.astype(np.float32)


def make_soft_lowpass(size: int, cutoff_radius: float, order: int = 4) -> np.ndarray:
    """
    Build a smooth circular low-pass filter in the Fourier plane.

    Using a soft edge instead of an ideal binary circle helps reduce ringing
    in the generated phase surface.
    """
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    radius = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    cutoff = max(float(cutoff_radius), 1.0)
    return np.exp(-((radius / cutoff) ** order)).astype(np.float32)


def generate_bandlimited_surface(
    size: int = 1024,
    cutoff_radius: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a smooth zero-mean random surface with controlled spatial bandwidth.

    The returned surface is standardized to unit variance so phase amplitudes can
    be tuned directly in radians later.
    """
    rng = rng or np.random.default_rng()
    noise = rng.standard_normal((size, size)).astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(noise))
    lowpass = make_soft_lowpass(size=size, cutoff_radius=cutoff_radius)
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * lowpass)).real
    filtered = filtered.astype(np.float32)
    filtered -= filtered.mean()
    filtered /= (filtered.std() + 1e-8)
    return filtered


def generate_smooth_phase_fourier(
    size: int = 1024,
    cutoff_radius: float = 20.0,
    phase_std: float = np.pi,
    seed: int | None = None,
    wrap_output: bool = False,
    phase_span: float = 0.75,
    softness: float = 1.8,
) -> np.ndarray:
    """
    Generate a smooth random SLM modulation.

    By default this returns smoothed random noise instead of wrapping the phase
    modulo 2π.  Set wrap_output=True to recover the old wrapped behavior.
    """
    rng = np.random.default_rng(seed)
    surface = generate_bandlimited_surface(size=size, cutoff_radius=cutoff_radius, rng=rng)
    if not wrap_output:
        return percentile_normalize_to_unit(surface)

    phase = phase_std * surface
    return wrap_phase_to_unit(phase)


def generate_quadratic_focus_phase(size: int = 1024, focus_cycles: float = 24.0) -> np.ndarray:
    """
    Generate a lens-like quadratic phase.

    focus_cycles controls how many 2π wraps appear from the center to the edge.
    Larger values pull energy back toward the optical axis more strongly.
    """
    _, r2 = radial_grid(size)
    r2 /= max(r2.max(), 1.0)
    return (TWO_PI * focus_cycles * r2).astype(np.float32)


def generate_wrapped_quadratic_lens_unit(
    size: int = 1024,
    focus_cycles: float = 24.0,
    invert: bool = False,
) -> np.ndarray:
    """
    Generate a wrapped quadratic lens phase in [0, 1).

    focus_cycles is the number of 2π wraps from center to the corner.  Set
    invert=True to flip the lens sign.
    """
    phase = generate_quadratic_focus_phase(size=size, focus_cycles=focus_cycles)
    if invert:
        phase = -phase
    return wrap_phase_to_unit(phase)


def generate_center_concentrating_phase_pair(
    size: int = 1024,
    layer1_cutoff: float = 18.0,
    layer2_cutoff: float = 42.0,
    layer1_rand_strength: float = 0.80 * np.pi,
    layer2_rand_strength: float = 0.10 * np.pi,
    conjugate_strength: float = 0.60,
    focus_cycles_layer1: float = 0.0,
    focus_cycles_layer2: float = 24.0,
    edge_taper_radius: float = 0.82,
    smooth_modulation: bool = True,
    layer1_phase_span: float = 0.92,
    layer2_phase_span: float = 0.88,
    softness: float = 1.05,
    focus_span_layer2: float = 0.22,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a phase-mask pair for "scatter first, refocus later".

    Design idea:
    - layer1: band-limited random phase to scramble the input field
    - layer2: weaker independent random phase
              - partial conjugate of layer1's random component
              + quadratic focus phase to pull energy back to the center

    The radial taper suppresses very strong random phase near the aperture edge,
    which is where peripheral lobes are easiest to excite.
    """
    rng = np.random.default_rng(seed)
    surf1 = generate_bandlimited_surface(size=size, cutoff_radius=layer1_cutoff, rng=rng)
    surf2 = generate_bandlimited_surface(size=size, cutoff_radius=layer2_cutoff, rng=rng)

    r, _ = radial_grid(size)
    taper = np.exp(-((r / max(edge_taper_radius, 1e-3)) ** 8)).astype(np.float32)

    layer1_rand = layer1_rand_strength * taper * surf1
    layer2_rand = layer2_rand_strength * taper * surf2

    focus1 = generate_quadratic_focus_phase(size=size, focus_cycles=focus_cycles_layer1)
    focus2 = generate_quadratic_focus_phase(size=size, focus_cycles=focus_cycles_layer2)

    layer1_phase = layer1_rand + focus1
    layer2_phase = (-conjugate_strength * layer1_rand) + layer2_rand + focus2

    if smooth_modulation:
        # Keep this path as a smooth, non-wrapped modulation path for callers,
        # but the main script below now uses direct smoothed noise.
        layer1_unit = percentile_normalize_to_unit(taper * surf1)
        layer2_surface = (
            -conjugate_strength * taper * surf1
            + (layer2_rand_strength / max(layer1_rand_strength, 1e-8)) * taper * surf2
            + focus_span_layer2 * normalized_focus_surface(size)
        )
        layer2_unit = percentile_normalize_to_unit(layer2_surface)
        return layer1_unit, layer2_unit

    return wrap_phase_to_unit(layer1_phase), wrap_phase_to_unit(layer2_phase)


def save_phase(path: str, phase_unit: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    phase_unit = phase_unit.astype(np.float32)
    np.save(path, phase_unit)

    preview_path = os.path.splitext(path)[0] + ".png"
    preview = np.round(np.clip(phase_unit, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview).save(preview_path)


if __name__ == "__main__":
    out_dir = "slm_phase"
    os.makedirs(out_dir, exist_ok=True)

    # Baseline pair: generate 256x256 smooth phase, then bicubic-resize to 1024x1024.
    baseline_layer1 = generate_lowres_smoothed_noise_unit(lowres_size=256, target_size=1024, sigma=3.5, seed=42)
    baseline_layer2 = generate_lowres_smoothed_noise_unit(lowres_size=256, target_size=1024, sigma=2.0, seed=43)
    save_phase(os.path.join(out_dir, "phase_layer1_narrow.npy"), baseline_layer1)
    save_phase(os.path.join(out_dir, "phase_layer2_wide.npy"), baseline_layer2)

    wrapped_lens = generate_wrapped_quadratic_lens_unit(size=1024, focus_cycles=24.0)
    save_phase(os.path.join(out_dir, "phase_wrapped_quadratic_lens.npy"), wrapped_lens)

    # Recommended pair: scatter is smoothed noise; refocus keeps a smooth
    # quadratic focusing trend on top of smoothed noise.
    refocus_layer1 = generate_lowres_smoothed_noise_unit(lowres_size=256, target_size=1024, sigma=2.5, seed=44)
    # refocus_noise = generate_lowres_smoothed_noise_unit(lowres_size=256, target_size=1024, sigma=1.5, seed=45)
    # refocus_layer2 = add_smooth_focus_to_noise(refocus_noise, focus_strength=0.5)
    # refocus_layer2 = generate_wrapped_quadratic_lens_unit(size=1024)
    refocus_layer2 = generate_lowres_smoothed_noise_unit(lowres_size=256, target_size=1024, sigma=2.5, seed=48)
    save_phase(os.path.join(out_dir, "phase_layer1_scatter.npy"), refocus_layer1)
    save_phase(os.path.join(out_dir, "phase_layer2_refocus.npy"), refocus_layer2)