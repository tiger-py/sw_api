from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import datetime as dt
import math
import time

import requests


# ============================================================
# Exceptions
# ============================================================

class OpenMeteoError(RuntimeError):
    """Raised when the Open-Meteo API cannot be queried or parsed."""


# ============================================================
# Internal helpers
# ============================================================

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Very small in-memory cache to avoid hammering Open-Meteo while you drag sliders.
# Keyed by (lat, lon, tilt, azimuth, year)
_TIMESERIES_CACHE: Dict[Tuple[float, float, float, float, int], "TiltedTimeseries"] = {}


@dataclass
class MeteoTimeseries:
    """Hourly meteorological + radiative drivers in UTC.

    This is used for:
      - Water shortwave baseline/shaded estimation (needs DNI/DHI and solar zenith).
      - Audit pack export of hourly meteo series.
    """
    times_utc: List[dt.datetime]
    ghi_w_m2: List[float]
    dni_w_m2: List[float]
    dhi_w_m2: List[float]
    tair_c: List[float]
    rh_pct: List[float]
    wind_m_s: List[float]


_METEO_CACHE: Dict[Tuple[float, float, int], MeteoTimeseries] = {}


def get_meteo_timeseries(*, latitude: float, longitude: float, year: int) -> MeteoTimeseries:
    """Fetch hourly GHI/DNI/DHI + basic meteo from Open-Meteo.

    We intentionally request variables that are stable and widely available:
      - shortwave_radiation (GHI)
      - direct_radiation (DNI on horizontal projection; we combine with solar position in water model)
      - diffuse_radiation (DHI)
      - temperature_2m, relative_humidity_2m, wind_speed_10m

    Notes:
      - Returned times are naive datetime in UTC (tzinfo=None) to match the rest of the codebase.
      - Results are cached in-process by (lat, lon, year).
    """
    key = (_round_coord(latitude), _round_coord(longitude), int(year))
    if key in _METEO_CACHE:
        return _METEO_CACHE[key]

    start_date = f"{int(year)}-01-01"
    end_date = f"{int(year)}-12-31"

    url_archive = "https://archive-api.open-meteo.com/v1/archive"
    url_forecast = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ]),
        "timezone": "UTC",
    }

    # Annual simulations require historical ranges, which the forecast endpoint often rejects.
    # Prefer the archive endpoint; fall back to forecast only if archive is unavailable.
    last_exc = None
    last_status = None
    last_text = None
    data = None
    for url in (url_archive, url_forecast):
        try:
            r = requests.get(url, params=params, timeout=60)
        except Exception as e:
            last_exc = e
            continue

        if r.status_code == 200:
            data = r.json()
            break

        last_status = r.status_code
        last_text = (r.text or "")[:200]

        # If forecast rejects the date range, try archive.
        if url == url_forecast and r.status_code == 400:
            continue

    if data is None:
        if last_exc is not None:
            raise OpenMeteoError(f"Open-Meteo unreachable: {type(last_exc).__name__}: {last_exc}")
        raise OpenMeteoError(f"Open-Meteo error {last_status}: {last_text}")
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        ts = MeteoTimeseries(
            times_utc=[],
            ghi_w_m2=[],
            dni_w_m2=[],
            dhi_w_m2=[],
            tair_c=[],
            rh_pct=[],
            wind_m_s=[],
        )
        _METEO_CACHE[key] = ts
        return ts

    def _as_list(name: str) -> List[float]:
        v = hourly.get(name)
        if v is None:
            return [0.0] * len(times)
        return [float(x) if x is not None else 0.0 for x in v]

    times_utc: List[dt.datetime] = []
    for t in times:
        # Open-Meteo UTC times are typically "YYYY-MM-DDTHH:MM" (no Z)
        try:
            times_utc.append(dt.datetime.fromisoformat(str(t)))
        except Exception:
            # Fallback: strip timezone if present
            tt = str(t).replace("Z", "")
            times_utc.append(dt.datetime.fromisoformat(tt))

    ts = MeteoTimeseries(
        times_utc=times_utc,
        ghi_w_m2=_as_list("shortwave_radiation"),
        dni_w_m2=_as_list("direct_radiation"),
        dhi_w_m2=_as_list("diffuse_radiation"),
        tair_c=_as_list("temperature_2m"),
        rh_pct=_as_list("relative_humidity_2m"),
        wind_m_s=_as_list("wind_speed_10m"),
    )
    _METEO_CACHE[key] = ts
    return ts


def solar_position_approx_utc(*, t_utc: dt.datetime, latitude: float, longitude: float) -> Tuple[float, float]:
    """Approximate solar zenith/azimuth (radians) for a UTC timestamp.

    Returns:
      (zenith_rad, azimuth_rad) where azimuth is measured clockwise from North.

    This is a compact NOAA-style approximation suitable for energy modeling.
    """
    # Convert to Julian day
    # Treat t_utc as UTC naive datetime.
    y, m, d = t_utc.year, t_utc.month, t_utc.day
    hr = t_utc.hour + t_utc.minute / 60.0 + t_utc.second / 3600.0

    if m <= 2:
        y -= 1
        m += 12
    A = math.floor(y / 100)
    B = 2 - A + math.floor(A / 4)
    JD = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + B - 1524.5 + hr / 24.0
    T = (JD - 2451545.0) / 36525.0

    # Solar coordinates
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)

    Mrad = math.radians(M)
    C = (1.914602 - T * (0.004817 + 0.000014 * T)) * math.sin(Mrad)         + (0.019993 - 0.000101 * T) * math.sin(2 * Mrad)         + 0.000289 * math.sin(3 * Mrad)
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lambda_sun = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Obliquity
    eps0 = 23.0 + (26.0 + ((21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60.0)) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))

    # Declination
    decl = math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(lambda_sun)))

    # Equation of time (minutes)
    yterm = math.tan(math.radians(eps) / 2.0)
    yterm *= yterm
    EoT = 4.0 * math.degrees(
        yterm * math.sin(2 * math.radians(L0))
        - 2 * e * math.sin(Mrad)
        + 4 * e * yterm * math.sin(Mrad) * math.cos(2 * math.radians(L0))
        - 0.5 * yterm * yterm * math.sin(4 * math.radians(L0))
        - 1.25 * e * e * math.sin(2 * Mrad)
    )

    # True solar time (minutes)
    # longitude in degrees, positive east.
    time_offset = EoT + 4.0 * longitude
    tst = (t_utc.hour * 60.0 + t_utc.minute + t_utc.second / 60.0 + time_offset) % 1440.0

    # Hour angle
    ha = math.radians(tst / 4.0 - 180.0)

    lat_rad = math.radians(latitude)
    # Solar zenith
    cos_zen = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zen = math.acos(cos_zen)

    # Azimuth (clockwise from North)
    sin_az = -math.sin(ha) * math.cos(decl) / max(1e-12, math.sin(zen))
    cos_az = (math.sin(decl) - math.sin(lat_rad) * math.cos(zen)) / max(1e-12, (math.cos(lat_rad) * math.sin(zen)))
    az = math.atan2(sin_az, cos_az)
    # atan2 gives (-pi, pi) from -x? adjust to [0, 2pi)
    if az < 0:
        az += 2 * math.pi

    return zen, az

@dataclass
class TiltedTimeseries:
    """
    Hourly plane-of-array irradiance on a given tilted surface.

    All irradiances are W/m², at 1-hour resolution from Open-Meteo
    (global_tilted_irradiance).
    """
    times: List[dt.datetime]
    poa_w_m2: List[float]

    # Pre-computed aggregates for convenience
    total_poa_sum_wm2: float
    count: int
    monthly_poa_sum_wm2: Dict[str, float]  # "YYYY-MM" -> sum of W/m² over all hours in that month

def _normalize_azimuth_deg(deg: float) -> float:
    """
    Map any azimuth in degrees to the range [-180, 180].

    UI slider uses 0°=N, 90°=E, 180°=S, 270°=W, 360°=N.
    Open-Meteo requires azimuth in [-180, 180], same reference.

    Examples:
      0   ->   0
      90  ->  90
      180 -> 180
      270 -> -90
      210 -> -150
      -190 -> 170
    """
    # Wrap into [0, 360)
    d = float(deg) % 360.0
    # Shift to [-180, 180)
    if d > 180.0:
        d -= 360.0
    return d

def _round_coord(x: float, digits: int = 3) -> float:
    return float(round(x, digits))


def _round_angle(x: float, digits: int = 1) -> float:
    return float(round(x, digits))


def _fetch_tilted_timeseries(
    latitude: float,
    longitude: float,
    tilt_deg: float,
    azimuth_deg: float,
    year: int | None = None,
) -> TiltedTimeseries:
    """
    Fetch hourly global_tilted_irradiance for a full year from Open-Meteo.

    Units from Open-Meteo:
        global_tilted_irradiance: W/m² (instantaneous / hourly mean)

    We assume 1-hour steps, so annual energy per m² is:
        E_kWh_per_m2 = sum(G_tilted_Wm2) * 1h / 1000
    """

    if year is None:
        # Use last full year, to avoid partial current year artifacts
        today = dt.date.today()
        year = today.year - 1

    key = (
        _round_coord(latitude),
        _round_coord(longitude),
        _round_angle(tilt_deg),
        _round_angle(azimuth_deg),
        int(year),
    )

    if key in _TIMESERIES_CACHE:
        return _TIMESERIES_CACHE[key]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "global_tilted_irradiance",
        "tilt": tilt_deg,
        "azimuth": azimuth_deg,
        "timezone": "UTC",
    }

    last_exc: Exception | None = None
    for _attempt in (1, 2):
        try:
            resp = requests.get(_ARCHIVE_URL, params=params, timeout=(4, 20))
            if resp.status_code != 200:
                # Bubble up detailed message for debugging
                raise OpenMeteoError(
                    f"Open-Meteo HTTP {resp.status_code}: {resp.text[:200]}"
                )

            data = resp.json()
            hourly = data.get("hourly") or {}
            times_str = hourly.get("time") or []
            poa_vals = hourly.get("global_tilted_irradiance") or []

            if not times_str or not poa_vals or len(times_str) != len(poa_vals):
                raise OpenMeteoError("Missing or inconsistent hourly global_tilted_irradiance")

            times: List[dt.datetime] = []
            poa_w_m2: List[float] = []
            total_sum = 0.0
            monthly_sum: Dict[str, float] = {}

            for t_str, g in zip(times_str, poa_vals):
                try:
                    g_val = float(g)
                except (TypeError, ValueError):
                    continue

                try:
                    t_dt = dt.datetime.fromisoformat(t_str)
                except Exception:
                    # Should not happen, but don't crash whole sim if one timestamp is bad
                    continue

                times.append(t_dt)
                poa_w_m2.append(g_val)
                total_sum += g_val

                month_key = f"{t_dt.year:04d}-{t_dt.month:02d}"
                monthly_sum[month_key] = monthly_sum.get(month_key, 0.0) + g_val

            if not times:
                raise OpenMeteoError("No valid hourly samples after parsing Open-Meteo response")

            ts = TiltedTimeseries(
                times=times,
                poa_w_m2=poa_w_m2,
                total_poa_sum_wm2=total_sum,
                count=len(times),
                monthly_poa_sum_wm2=monthly_sum,
            )

            _TIMESERIES_CACHE[key] = ts
            return ts

        except Exception as e:
            last_exc = e
            # Small backoff in case of transient issues
            time.sleep(0.3)

    if isinstance(last_exc, OpenMeteoError):
        raise last_exc
    raise OpenMeteoError(f"Open-Meteo unreachable: {type(last_exc).__name__}: {last_exc}")


def get_tilted_timeseries(
    *,
    latitude: float,
    longitude: float,
    tilt_deg: float,
    azimuth_deg: float,
    year: int | None = None,
) -> TiltedTimeseries:
    """
    Public wrapper so api.py can fetch the raw hourly POA series for a single orientation.
    """
    return _fetch_tilted_timeseries(
        latitude=latitude,
        longitude=longitude,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        year=year,
    )


# ============================================================
# Public simulation API
# ============================================================

def simulate_annual_output(
    *,
    latitude: float,
    longitude: float,
    surface_tilt_deg: float,
    surface_azimuths_deg: Sequence[float],
    panels_per_azimuth: Sequence[int],
    panel_area_m2: float,
    eff_stc: float,
    noct: float,
    temp_coeff: float,
    cooling_offset: float,
    year: int | None = None,
) -> Dict[str, object]:
    """
    Core simulation entry point used by api.py.

    For each distinct surface azimuth, we:
      1) Query Open-Meteo for hourly global_tilted_irradiance (W/m²).
      2) Convert irradiance -> DC energy, assuming a simple efficiency model.
      3) Aggregate annual and monthly energies.

    NOTE:
    - We intentionally keep the temperature model very simple here, to stabilise
      the scaling. We can refine it later (NOCT, temp_coeff) once the base
      behaviour is validated end-to-end again.
    """

    if len(surface_azimuths_deg) != len(panels_per_azimuth):
        raise ValueError("surface_azimuths_deg and panels_per_azimuth must have the same length")

    # Ensure deterministic order of monthly keys
    monthly_energy_kwh: Dict[str, float] = {}

    total_annual_energy_kwh = 0.0
    total_mean_poa_weighted = 0.0

    total_panels = max(1, int(sum(int(n) for n in panels_per_azimuth)))

    for az_deg, n_panels in zip(surface_azimuths_deg, panels_per_azimuth):
        n_panels = int(n_panels)
        if n_panels <= 0:
            continue

        # Map our UI convention (0=N, 90=E, 180=S, 270=W, 360≡0)
        # into the -180…+180° azimuth expected by Open-Meteo.
        az_norm = _normalize_azimuth_deg(float(az_deg))

        ts = _fetch_tilted_timeseries(
            latitude=latitude,
            longitude=longitude,
            tilt_deg=surface_tilt_deg,
            azimuth_deg=az_norm,
            year=year,
        )
        
        # Mean POA for this orientation (W/m²)
        mean_poa_this = ts.total_poa_sum_wm2 / ts.count

        # --- Energy calculation ---
        # Annual energy per m² on module plane (kWh/m²)
        annual_poa_kwh_per_m2 = ts.total_poa_sum_wm2 / 1000.0

        # Simple constant-efficiency DC model at STC (we can layer temp losses later)
        annual_dc_kwh_per_panel = annual_poa_kwh_per_m2 * panel_area_m2 * eff_stc

        annual_dc_kwh_for_orientation = annual_dc_kwh_per_panel * n_panels
        total_annual_energy_kwh += annual_dc_kwh_for_orientation

        # Monthly breakdown
        for month_key, month_sum_wm2 in ts.monthly_poa_sum_wm2.items():
            month_kwh_per_m2 = month_sum_wm2 / 1000.0
            month_kwh_per_panel = month_kwh_per_m2 * panel_area_m2 * eff_stc
            month_kwh_for_orientation = month_kwh_per_panel * n_panels

            monthly_energy_kwh[month_key] = monthly_energy_kwh.get(month_key, 0.0) + month_kwh_for_orientation

        # Contribution to overall mean POA, weighted by panel count
        total_mean_poa_weighted += mean_poa_this * n_panels

    # Panel-weighted mean POA across all orientations
    mean_poa_w_m2 = total_mean_poa_weighted / total_panels

    # Sort monthly keys for nicer JSON
    monthly_energy_kwh_sorted = dict(sorted(monthly_energy_kwh.items()))

    return {
        "annual_energy_kwh": float(total_annual_energy_kwh),
        "monthly_energy_kwh": {k: float(v) for k, v in monthly_energy_kwh_sorted.items()},
        "mean_poa_w_m2": float(mean_poa_w_m2),
    }


# ============================================================
# Water-plane irradiance metrics from GPU shading samples
# ============================================================

def _nearest_time_index(times: Sequence[dt.datetime], target: dt.datetime) -> int:
    lo, hi = 0, len(times) - 1
    if hi <= 0:
        return 0
    if target <= times[0]:
        return 0
    if target >= times[-1]:
        return hi
    while lo <= hi:
        mid = (lo + hi) // 2
        tmid = times[mid]
        if tmid < target:
            lo = mid + 1
        elif tmid > target:
            hi = mid - 1
        else:
            return mid
    i1 = max(0, min(len(times) - 1, lo))
    i0 = max(0, i1 - 1)
    return i0 if (target - times[i0]) <= (times[i1] - target) else i1


def water_metrics_from_shading_samples(
    *,
    latitude: float,
    longitude: float,
    year: int,
    samples: Sequence[dict],
    svf: float = 1.0,
) -> Dict[str, float]:
    ts = get_meteo_timeseries(latitude=latitude, longitude=longitude, year=year)
    if not ts.times_utc:
        return {"water_baseline_kwh_m2": 0.0, "water_shaded_kwh_m2": 0.0, "water_reduction_pct": 0.0}

    baseline = 0.0
    shaded = 0.0

    for s in samples:
        if not isinstance(s, dict):
            continue
        t = s.get("time_utc") or s.get("t_utc") or s.get("time")
        tau = float(s.get("tau", 1.0))
        wh = float(s.get("weight_hours", 0.0))
        if wh <= 0:
            continue

        t_dt = None
        if isinstance(t, dt.datetime):
            t_dt = t
        elif isinstance(t, str) and t:
            try:
                t_dt = dt.datetime.fromisoformat(t.replace("Z", ""))
            except Exception:
                t_dt = None
        if t_dt is None:
            continue
        if t_dt.tzinfo is not None:
            t_dt = t_dt.astimezone(dt.timezone.utc).replace(tzinfo=None)

        i = _nearest_time_index(ts.times_utc, t_dt)
        dni = ts.dni_w_m2[i]
        dhi = ts.dhi_w_m2[i]
        if dni is None or dhi is None:
            continue

        zen, _az = solar_position_approx_utc(t_utc=ts.times_utc[i], latitude=latitude, longitude=longitude)
        if zen >= math.pi / 2:
            continue
        cosz = max(0.0, math.cos(zen))

        i0 = float(dni) * cosz + float(dhi)
        svf_clamped = max(0.0, min(1.0, float(svf)))
        i1 = float(dni) * cosz * max(0.0, min(1.0, tau)) + float(dhi) * svf_clamped

        baseline += i0 * wh / 1000.0
        shaded += i1 * wh / 1000.0

    red = 0.0 if baseline <= 1e-9 else max(0.0, min(100.0, 100.0 * (1.0 - shaded / baseline)))
    return {
        "water_baseline_kwh_m2": float(baseline),
        "water_shaded_kwh_m2": float(shaded),
        "water_reduction_pct": float(red),
    }
