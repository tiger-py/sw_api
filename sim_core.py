from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Any
import datetime as dt
import math
import time
import os

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

# Daily precipitation cache: (lat, lon, year) -> list[mm/day]
_PRECIP_CACHE: Dict[Tuple[float, float, int], List[float]] = {}

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

def _wrap_deg_180(az_deg: float) -> float:
    """Wrap degrees to (-180, 180]."""
    a = float(az_deg) % 360.0
    if a > 180.0:
        a -= 360.0
    return a

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

def _normalize_azimuth_deg(deg_ui: float) -> float:
    """
    Convert UI azimuth (0=N,90=E,180=S,270=W) into Open-Meteo azimuth
    where 0=S, -90=E, +90=W, ±180=N, and clamp to [-180, 180).

    Mapping:
      UI 180 (South) -> 0
      UI  90 (East)  -> -90
      UI 270 (West)  -> +90
      UI   0 (North) -> -180 (or +180)

    Note: For roof/canopy arrays, we apply the same mapping to maintain
    consistency with the frontend orientation.
    """
    # shift UI so that South becomes 0 (Open-Meteo convention)
    d = (float(deg_ui) - 180.0) % 360.0
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

def debug_compare_openmeteo_azimuth_conventions(*, latitude: float, longitude: float, tilt_deg: float, year: int) -> Dict[str, float]:
    """
    Compare annual POA sums for the SAME physical orientations using two azimuth conventions:
      A) passthrough: assumes Open-Meteo expects 0=N,90=E,180=S,270=W
      B) normalized:  assumes Open-Meteo expects 0=S, -90=E, +90=W, ±180=N  (your _normalize_azimuth_deg)
    Returns annual kWh/m^2 proxies (sum(W/m²)/1000).
    """
    def annual_kwh_m2(az_for_query: float) -> float:
        ts = _fetch_tilted_timeseries(
            latitude=latitude,
            longitude=longitude,
            tilt_deg=float(tilt_deg),
            azimuth_deg=_wrap_deg_180(float(az_for_query)),
            year=int(year),
        )
        return float(ts.total_poa_sum_wm2) / 1000.0

    # UI meanings
    ui_north = 0.0
    ui_south = 180.0
    ui_east  = 90.0
    ui_west  = 270.0

    # Convention A: pass-through
    A = {
        "A_pass_north": annual_kwh_m2(ui_north),
        "A_pass_south": annual_kwh_m2(ui_south),
        "A_pass_east":  annual_kwh_m2(ui_east),
        "A_pass_west":  annual_kwh_m2(ui_west),
    }

    # Convention B: your normalization
    B = {
        "B_norm_north": annual_kwh_m2(_normalize_azimuth_deg(ui_north)),
        "B_norm_south": annual_kwh_m2(_normalize_azimuth_deg(ui_south)),
        "B_norm_east":  annual_kwh_m2(_normalize_azimuth_deg(ui_east)),
        "B_norm_west":  annual_kwh_m2(_normalize_azimuth_deg(ui_west)),
    }

    out = {}
    out.update(A)
    out.update(B)
    return out


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

        # For roof arrays with single orientation, log the conversion
        if len(surface_azimuths_deg) == 1 and os.environ.get("SW_DEBUG_AZ", "") == "1":
            print("[ROOF AZ CONVERSION]", {
                "ui_az": float(az_deg),
                "openmeteo_az": az_norm,
                "tilt": surface_tilt_deg
            })

        if os.environ.get("SW_DEBUG_AZ", "") == "1":
            print("[POA QUERY]", {
                "lat": float(latitude),
                "lon": float(longitude),
                "tilt_deg": float(surface_tilt_deg),
                "ui_or_model_az_deg": float(az_deg),
                "openmeteo_az_deg": float(az_norm),
                "n_panels": int(n_panels),
            })

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

def validate_base_array_simulation(
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
) -> Dict[str, Any]:
    """Validate base array simulation for potential flaws."""
    
    diagnostics = {
        "inputs_validated": True,
        "warnings": [],
        "errors": [],
        "sanity_checks": {},
        "expected_ranges": {},
        "edge_cases": [],
        "suggestions": []
    }
    
    # 1. Input Validation
    if not (-90 <= latitude <= 90):
        diagnostics["errors"].append(f"Invalid latitude: {latitude}")
        diagnostics["inputs_validated"] = False
    
    if not (-180 <= longitude <= 180):
        diagnostics["errors"].append(f"Invalid longitude: {longitude}")
        diagnostics["inputs_validated"] = False
    
    if not (0 <= surface_tilt_deg <= 90):
        diagnostics["warnings"].append(f"Unusual tilt angle: {surface_tilt_deg}° (typical range 0-90°)")
   
    if not (0 <= sum(panels_per_azimuth) <= 10000):
        diagnostics["warnings"].append(f"Unusual panel count: {sum(panels_per_azimuth)}")
    
    if not (0.5 <= panel_area_m2 <= 5.0):
        diagnostics["warnings"].append(f"Unusual panel area: {panel_area_m2} m² (typical 1.0-2.5 m²)")
    
    if not (0.05 <= eff_stc <= 0.30):
        diagnostics["warnings"].append(f"Unusual STC efficiency: {eff_stc} (typical 0.15-0.25)")
    
    if not (30 <= noct <= 60):
        diagnostics["warnings"].append(f"Unusual NOCT: {noct}°C (typical 40-50°C)")
    
    if not (-0.01 >= temp_coeff >= -0.002):
        diagnostics["warnings"].append(f"Unusual temperature coefficient: {temp_coeff} (typical -0.005 to -0.003)")
    
    # 2. Orientation Balance Check
    if len(surface_azimuths_deg) == 2:
        az1, az2 = surface_azimuths_deg
        diff = (az2 - az1) % 360
        if not (175 <= diff <= 185):
            diagnostics["warnings"].append(
                f"Waves array orientations not opposing: {az1:.1f}° vs {az2:.1f}° (diff {diff:.1f}°)"
            )
    
    # 3. Panel Distribution Check
    total_panels = sum(panels_per_azimuth)
    if total_panels == 0:
        diagnostics["errors"].append("Zero total panels")
        diagnostics["inputs_validated"] = False
    else:
        for i, n in enumerate(panels_per_azimuth):
            if n < 0:
                diagnostics["errors"].append(f"Negative panel count for orientation {i}")
                diagnostics["inputs_validated"] = False
    
    # 4. Fetch small sample to validate Open-Meteo connectivity
    try:
        test_ts = _fetch_tilted_timeseries(
            latitude=latitude,
            longitude=longitude,
            tilt_deg=surface_tilt_deg,
            azimuth_deg=_normalize_azimuth_deg(float(surface_azimuths_deg[0])),
            year=2024,
        )
        
        if test_ts.count > 0:
            max_poa = max(test_ts.poa_w_m2)
            mean_poa = test_ts.total_poa_sum_wm2 / test_ts.count
            
            diagnostics["sanity_checks"]["openmeteo_data_available"] = True
            diagnostics["sanity_checks"]["max_poa_w_m2"] = float(max_poa)
            diagnostics["sanity_checks"]["mean_poa_w_m2"] = float(mean_poa)
            diagnostics["sanity_checks"]["data_points"] = test_ts.count
            
            if max_poa > 1500:
                diagnostics["warnings"].append(f"Suspiciously high POA: {max_poa:.0f} W/m²")
            if mean_poa < 50:
                diagnostics["warnings"].append(f"Very low mean POA: {mean_poa:.0f} W/m² (possible polar night/latitude issue)")
            
        else:
            diagnostics["errors"].append("No data returned from Open-Meteo")
            diagnostics["sanity_checks"]["openmeteo_data_available"] = False
            
    except Exception as e:
        diagnostics["errors"].append(f"Open-Meteo connection failed: {str(e)}")
        diagnostics["sanity_checks"]["openmeteo_data_available"] = False
    
    # 5. Expected Ranges for Outputs
    diagnostics["expected_ranges"] = {
        "annual_energy_kwh_per_panel": {
            "min": 150,
            "max": 800,
            "typical": "300-500"
        },
        "mean_poa_w_m2": {
            "min": 100,
            "max": 400,
            "typical": "200-350"
        }
    }
    
    # 6. Edge Cases Detection
    if abs(latitude) > 66.5:
        diagnostics["edge_cases"].append("Polar region - seasonal extremes expected")
    
    if surface_tilt_deg > 80:
        diagnostics["edge_cases"].append("Near-vertical array - unusual mounting")
    
    if total_panels < 5:
        diagnostics["edge_cases"].append("Very small array - statistics may be noisy")
    
    # 7. Suggestions
    if diagnostics["warnings"]:
        diagnostics["suggestions"].append("Review warnings before scaling results")
    
    if diagnostics["errors"]:
        diagnostics["suggestions"].append("Fix errors in base configuration")
    
    return diagnostics

def _parse_month_key_from_time_utc(time_utc: str) -> str | None:
    if not time_utc or len(time_utc) < 7:
        return None
    try:
        return time_utc[0:7]
    except Exception:
        return None

def pv_energy_from_tau_samples(*, base: Dict[str, object], pv_samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    monthly_energy = base.get("monthly_energy_kwh", {}) or {}
    if not isinstance(monthly_energy, dict):
        monthly_energy = {}

    tau_wsum: Dict[str, float] = {}
    tau_tsum: Dict[str, float] = {}

    wsum_all = 0.0
    tsum_all = 0.0

    for s in (pv_samples or []):
        try:
            time_utc = str(s.get("time_utc", ""))

            # Correct priority:
            #   1) tau_shadow (pure shading mask)
            #   2) tau (legacy)
            #   3) f_beam (incidence-weighted availability) only if nothing else exists
            tau_raw = s.get("tau_shadow", None)
            if tau_raw is None:
                tau_raw = s.get("tau", None)
            if tau_raw is None:
                tau_raw = s.get("f_beam", 1.0)

            tau = float(tau_raw)
            w = float(s.get("weight_hours", 0.0))
        except Exception:
            continue

        if w <= 0:
            continue

        tau = max(0.0, min(1.0, tau))
        mk = _parse_month_key_from_time_utc(time_utc)
        if not mk:
            continue

        tau_wsum[mk] = tau_wsum.get(mk, 0.0) + w
        tau_tsum[mk] = tau_tsum.get(mk, 0.0) + w * tau

        wsum_all += w
        tsum_all += w * tau

    tau_pv_monthly: Dict[str, float] = {}
    for mk, wsum in tau_wsum.items():
        if wsum > 0:
            tau_pv_monthly[mk] = tau_tsum.get(mk, 0.0) / wsum

    tau_pv_mean = (tsum_all / wsum_all) if wsum_all > 0 else 1.0
    tau_pv_mean = max(0.0, min(1.0, tau_pv_mean))

    monthly_shaded: Dict[str, float] = {}
    annual_shaded = 0.0

    for mk, e in monthly_energy.items():
        try:
            e0 = float(e)
        except Exception:
            continue

        tau_m = tau_pv_monthly.get(mk, tau_pv_mean)
        tau_m = max(0.0, min(1.0, tau_m))

        e1 = e0 * tau_m
        monthly_shaded[mk] = float(e1)
        annual_shaded += float(e1)

    return {
        "annual_energy_kwh_shaded_pv": float(annual_shaded),
        "monthly_energy_kwh_shaded_pv": dict(sorted(monthly_shaded.items())),
        "tau_pv_mean": float(tau_pv_mean),
        "tau_pv_monthly": dict(sorted(tau_pv_monthly.items())),
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


# ============================================================
# Enhanced PV Output Models (Hybrid Approach)
# ============================================================

def pvwatts_v5_model(
    poa_w_m2: float,
    t_amb_c: float,
    wind_speed_m_s: float,
    panel_area_m2: float,
    eff_stc: float,
    gamma_pmp: float,
    noct: float,
    system_losses: float = 0.14
) -> float:
    """
    NREL PVWatts v5 model with temperature correction
    Returns AC power in Watts
    """
     # Cell temperature model from PVWatts v5
    # At STC (1000 W/m², 25°C, 1 m/s), t_cell should be ~45-50°C for NOCT=45
    # The formula uses NOCT as reference point
    t_cell = t_amb_c + (poa_w_m2 / 1000) * (noct - 20) * (1 - 0.03 * wind_speed_m_s)
     
    
    # DC power with temperature derate
    p_dc_stc = poa_w_m2 * panel_area_m2 * eff_stc
    temp_derate = 1 + gamma_pmp * (t_cell - 25)
    temp_derate = max(0.5, min(1.05, temp_derate))
    
    # Low-light correction (PV modules perform worse at low light)
    low_light_factor = 1 - 0.02 * math.exp(-poa_w_m2 / 200)
    
    p_dc = p_dc_stc * temp_derate * low_light_factor
    
    # Apply system losses (inverter, wiring, soiling)
    p_ac = p_dc * (1 - system_losses)
    
    return max(0, p_ac)


def incidence_angle_modifier(incidence_angle_deg: float) -> float:
    """
    ASHRAE standard incidence angle modifier for PV modules
    """
    if incidence_angle_deg > 90:
        return 0
    theta = math.radians(incidence_angle_deg)
    return 1 - 0.05 * (1/max(0.01, math.cos(theta)) - 1)


def calculate_pv_output_optimized(
    *,
    latitude: float,
    longitude: float,
    year: int,
    tilt_deg: float,
    azimuth_deg: float,
    panel_area_m2: float,
    eff_stc: float,
    gamma_pmp: float,
    noct: float,
    system_losses: float = 0.14,
    accuracy: str = "balanced"
) -> Dict[str, float]:
    """
    Multi-tier PV output calculation
    
    Args:
        accuracy: "fast" (annual aggregation with avg temp),
                 "balanced" (monthly aggregation),
                 "accurate" (hourly with thermal model)
    """
    # Get POA timeseries
    ts = get_tilted_timeseries(
        latitude=latitude,
        longitude=longitude,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        year=year
    )
    
    # Get temperature/wind data
    meteo = get_meteo_timeseries(latitude=latitude, longitude=longitude, year=year)
    
    if accuracy == "fast":
        # Annual aggregation with average temperature correction
        annual_poa_kwh_m2 = ts.total_poa_sum_wm2 / 1000
        t_avg = sum(meteo.tair_c) / len(meteo.tair_c) if meteo.tair_c else 15
        
        # Simple temperature correction
        temp_derate = 1 + gamma_pmp * (t_avg - 25)
        temp_derate = max(0.7, min(1.0, temp_derate))
        
        annual_kwh = annual_poa_kwh_m2 * panel_area_m2 * eff_stc * temp_derate
        annual_kwh *= (1 - system_losses)
        
        return {
            "annual_energy_kwh": annual_kwh,
            "method": "fast",
            "system_losses": system_losses
        }
        
    elif accuracy == "balanced":
        # Monthly aggregation (12 calculations)
        monthly_kwh = {}
        annual_kwh = 0
        
        for month in range(1, 13):
            month_indices = [i for i, t in enumerate(ts.times) if t.month == month]
            if not month_indices:
                continue
                
            # Monthly POA sum
            month_poa_sum = sum(ts.poa_w_m2[i] for i in month_indices)
            month_poa_kwh_m2 = month_poa_sum / 1000
            
            # Monthly average temperature and wind
            month_temps = [meteo.tair_c[i] for i in month_indices if i < len(meteo.tair_c)]
            month_winds = [meteo.wind_m_s[i] for i in month_indices if i < len(meteo.wind_m_s)]
            t_avg_month = sum(month_temps) / len(month_temps) if month_temps else 15
            wind_avg_month = sum(month_winds) / len(month_winds) if month_winds else 2
            
            # PVWatts model for monthly average
            # Use average POA for the month (not hourly)
            avg_poa = month_poa_sum / len(month_indices)
            
            month_kwh = pvwatts_v5_model(
                poa_w_m2=avg_poa,
                t_amb_c=t_avg_month,
                wind_speed_m_s=wind_avg_month,
                panel_area_m2=panel_area_m2,
                eff_stc=eff_stc,
                gamma_pmp=gamma_pmp,
                noct=noct,
                system_losses=system_losses
            ) * len(month_indices) / 1000  # Convert Wh to kWh
            
            monthly_kwh[f"{year}-{month:02d}"] = month_kwh
            annual_kwh += month_kwh
        
        return {
            "annual_energy_kwh": annual_kwh,
            "monthly_energy_kwh": monthly_kwh,
            "method": "balanced",
            "system_losses": system_losses
        }

# ============================================================
# Water evaporation model helpers (Ladybug-aligned)
# ============================================================

def annual_climate_means(ts: MeteoTimeseries) -> Dict[str, float]:
    """Return annual mean climate drivers from the Open-Meteo hourly series.

    Returns:
        dict with keys: Tavg_C, RH_dec, Wavg_m_s
    """
    def _safe_mean(arr: List[float]) -> float:
        vals = [float(v) for v in arr if v is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0

    tavg = _safe_mean(ts.tair_c)
    rh = _safe_mean(ts.rh_pct) / 100.0
    wavg = _safe_mean(ts.wind_m_s)
    rh = max(0.0, min(1.0, rh))
    return {"Tavg_C": tavg, "RH_dec": rh, "Wavg_m_s": wavg}

# Keep the original function for backward compatibility
def water_evaporation_lpy_from_kwh_m2(*args, **kwargs):
    """Legacy function - kept for backward compatibility
    Returns a dict with expected keys for older API clients
    """
    # Return a placeholder response with zeros
    # This function is no longer used in the main simulation path
    return {"evap_lpy": 0.0, "evap_lpy_wind": 0.0}

def _get_monthly_climate_data(ts: MeteoTimeseries, month: int) -> Dict[str, float]:
    """Extract monthly climate averages from timeseries"""
    month_indices = [i for i, t in enumerate(ts.times_utc) if t.month == month]
    if not month_indices:
        return {
            "Rs": 0,
            "Tavg": 15,
            "wind": 2,
            "RH": 0.5,
            "days": 30
        }
    
    # Calculate averages
    Rs = sum(ts.ghi_w_m2[i] for i in month_indices) / len(month_indices)
    Tavg = sum(ts.tair_c[i] for i in month_indices) / len(month_indices)
    wind = sum(ts.wind_m_s[i] for i in month_indices) / len(month_indices)
    RH = sum(ts.rh_pct[i] for i in month_indices) / len(month_indices) / 100
    
    # Days in month
    days = len(set(t.day for t in [ts.times_utc[i] for i in month_indices]))
    
    return {
        "Rs": Rs,
        "Tavg": Tavg,
        "wind": wind,
        "RH": RH,
        "days": days
    }

def _get_daily_climate_data(ts: MeteoTimeseries, day_of_year: int) -> Dict[str, float]:
    """Extract daily climate averages from timeseries"""
    # This is a placeholder - actual implementation would need
    # to handle day-of-year to actual date mapping
    return {
        "Rs": 200,  # Placeholder
        "Tavg": 20,
        "wind": 3,
        "RH": 0.6,
        "ea": 1.2,
        "es": 2.3
    }

def infer_wind_block_factor(*, array_type: str, height_m: float | None) -> float:
    """Heuristic wind exposure factor (0..1), where 1 means fully exposed to wind."""
    h = float(height_m) if height_m is not None and math.isfinite(height_m) else None

    at = (array_type or "").lower()
    if at not in ("waves", "roof"):
        at = "waves"

    if h is None:
        return 0.2 if at == "waves" else 0.8

    def lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    if at == "waves":
        # Stage 1: 0.3m -> 5m : 0.10 -> 0.60
        # Stage 2: 5m  -> 12m: 0.60 -> 1.00
        if h <= 5.0:
            h0, h1 = 0.3, 5.0
            t = clamp01((h - h0) / (h1 - h0))
            return float(lerp(0.10, 0.60, t))
        else:
            h0, h1 = 5.0, 12.0
            t = clamp01((h - h0) / (h1 - h0))
            return float(lerp(0.60, 1.00, t))

    else:
        # roof/canopy: more exposed at all heights
        # Stage 1: 0.5m -> 6m  : 0.40 -> 0.90
        # Stage 2: 6m  -> 12m : 0.90 -> 1.00
        if h <= 6.0:
            h0, h1 = 0.5, 6.0
            t = clamp01((h - h0) / (h1 - h0))
            return float(lerp(0.40, 0.90, t))
        else:
            h0, h1 = 6.0, 12.0
            t = clamp01((h - h0) / (h1 - h0))
            return float(lerp(0.90, 1.00, t))


# ============================================================
# Water-plane irradiance metrics from GPU shading samples
# ============================================================

def water_metrics_from_shading_samples(
    *,
    latitude: float,
    longitude: float,
    year: int,
    samples: Sequence[dict],
    svf: float = 1.0,
    elevation_m: float = 0.6,
    water_area_m2: float,  # NEW: required parameter
    panel_coverage: float,  # NEW: required parameter  
    wind_block_factor: float,  # NEW: required parameter
    accuracy: str = "balanced" 
) -> Dict[str, float]:
    """
    Estimate annual shortwave energy on water and evaporation savings.
    
    Returns:
      water_baseline_kwh_m2: annual baseline shortwave [kWh/m^2/yr]
      water_shaded_kwh_m2: annual shaded shortwave [kWh/m^2/yr]
      water_reduction_pct: percent reduction from baseline to shaded [%]
      evap_mm_yr: annual evaporation depth [mm/yr]
      evap_liters_yr: annual evaporation volume [liters/yr]
    """
    ts = get_meteo_timeseries(latitude=latitude, longitude=longitude, year=year)
    
    if not ts.times_utc:
        return {
            "water_baseline_kwh_m2": 0.0,
            "water_shaded_kwh_m2": 0.0,
            "water_reduction_pct": 0.0,
            "evap_mm_yr": 0.0,
            "evap_liters_yr": 0.0,
            "evap_method": accuracy
        }

    svf_clamped = max(0.0, min(1.0, float(svf)))

    # Precompute per-UTC-day direct/diffuse energy [kWh/m^2/day]
    day_energy: Dict[dt.date, Tuple[float, float]] = {}
    for i, t_utc in enumerate(ts.times_utc):
        dni = ts.dni_w_m2[i]
        dhi = ts.dhi_w_m2[i]
        if dni is None or dhi is None:
            continue
        try:
            zen, _az = solar_position_approx_utc(t_utc=t_utc, latitude=latitude, longitude=longitude)
        except Exception:
            continue
        if zen >= math.pi / 2:
            continue
        cosz = max(0.0, math.cos(zen))
        direct_kwh = float(dni) * cosz / 1000.0
        diffuse_kwh = float(dhi) / 1000.0

        d = t_utc.date()
        if d in day_energy:
            dd, df = day_energy[d]
            day_energy[d] = (dd + direct_kwh, df + diffuse_kwh)
        else:
            day_energy[d] = (direct_kwh, diffuse_kwh)

    baseline = 0.0
    shaded = 0.0

    for s in samples:
        if not isinstance(s, dict):
            continue
        t = s.get("time_utc") or s.get("t_utc") or s.get("time")
        tau = float(s.get("tau", 1.0))
        tau = max(0.0, min(1.0, tau))
        wh = float(s.get("weight_hours", 0.0))
        if wh <= 0:
            continue

        # Parse sample time
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

        # Use representative-day integration
        d = t_dt.date()
        dd_df = day_energy.get(d)
        if dd_df is None:
            continue

        direct_kwh_day, diffuse_kwh_day = dd_df
        days = wh / 24.0
        baseline += (direct_kwh_day + diffuse_kwh_day) * days
        shaded += (direct_kwh_day * tau + diffuse_kwh_day * svf_clamped) * days

    red = 0.0 if baseline <= 1e-9 else max(0.0, min(100.0, 100.0 * (1.0 - shaded / baseline)))

    evap_results = calculate_water_evaporation_optimized(
        latitude=latitude,
        longitude=longitude,
        year=year,
        water_area_m2=water_area_m2,
        panel_coverage=panel_coverage,
        wind_block_factor=wind_block_factor,
        accuracy=accuracy
    )
    
    # Combine with existing results
    return {
        "water_baseline_kwh_m2": float(baseline),
        "water_shaded_kwh_m2": float(shaded),
        "water_reduction_pct": float(red),
        "evap_mm_yr": evap_results.get("evap_mm_yr", 0),
        "evap_liters_yr": evap_results.get("evap_liters_yr", 0),
        "evap_method": accuracy
    }

# ============================================================
# Enhanced Water Evaporation Models (Hybrid Approach)
# ============================================================

def fao_penman_monteith_daily(
    Rn_MJ_m2_day: float,
    T_C: float,
    u2_m_s: float,
    es_kPa: float,
    ea_kPa: float,
    G_MJ_m2_day: float = 0.0
) -> float:
    """
    FAO-56 Penman-Monteith for open water (ET0 in mm/day)
    """
    # Slope of saturation vapor pressure
    delta = 4098 * es_kPa / (T_C + 237.3)**2
    
    # Psychrometric constant at sea level
    gamma = 0.665e-3 * 101.3
    
    # Aerodynamic resistance for open water
    ra = 208 / max(0.1, u2_m_s) if u2_m_s > 0 else 1000
    
    numerator = 0.408 * delta * (Rn_MJ_m2_day - G) + gamma * (900/(T_C+273)) * u2_m_s * (es_kPa - ea_kPa)
    denominator = delta + gamma * (1 + 0.34 * u2_m_s)
    
    return max(0, numerator / denominator)


def priestley_taylor_daily(
    Rn_MJ_m2_day: float,
    T_C: float,
    alpha: float = 1.26
) -> float:
    """
    Priestley-Taylor evaporation (mm/day) - radiation-driven only
    Alpha = 1.26 for open water (Priestley & Taylor, 1972)
    """
    delta = 4098 * (0.6108 * math.exp(17.27 * T_C / (T_C + 237.3))) / (T_C + 237.3)**2
    gamma = 0.665e-3 * 101.3
    
    return alpha * (delta / (delta + gamma)) * Rn_MJ_m2_day / 2.45


def de_bruin_keijman_daily(
    Rs_MJ_m2_day: float,  # Incoming shortwave
    T_C: float,
    u2_m_s: float,
    RH: float
) -> float:
    """
    De Bruin-Keijman formula specifically for open water evaporation
    """
    # Net radiation estimation (water albedo ~0.06)
    albedo = 0.06
    Rn = Rs_MJ_m2_day * (1 - albedo) - 110 / 1e6 * 86400  # Simplified net longwave
    
    # Saturation vapor pressure
    es = 0.6108 * math.exp(17.27 * T_C / (T_C + 237.3))
    ea = es * (RH / 100)
    
    # Delta and gamma
    delta = 4098 * es / (T_C + 237.3)**2
    gamma = 0.665e-3 * 101.3
    
    # Wind function (De Bruin, 1982)
    f_u = 2.6 * (1 + 0.54 * u2_m_s)
    
    # Combination equation
    E = (delta * Rn + gamma * f_u * (es - ea)) / (delta + gamma) / 2.45
    
    return max(0, E)


def calculate_water_evaporation_optimized(
    *,
    latitude: float,
    longitude: float,
    year: int,
    water_area_m2: float,
    panel_coverage: float,  # 0-1
    wind_block_factor: float,
    accuracy: str = "balanced"
) -> Dict[str, float]:
    """
    Multi-tier water evaporation calculation
    
    Args:
        accuracy: "fast" (annual Priestley-Taylor), 
                 "balanced" (monthly De Bruin-Keijman),
                 "accurate" (daily FAO Penman-Monteith)
    """
    ts = get_meteo_timeseries(latitude=latitude, longitude=longitude, year=year)
    
    if accuracy == "fast":
        # Annual means with Priestley-Taylor
        clim = annual_climate_means(ts)
        R_kwh = sum(ts.ghi_w_m2) / 1000  # Annual kWh/m²
        Rn_MJ_day = R_kwh * 3.6 / 365  # Daily net radiation
        
        E_mm = priestley_taylor_daily(
            Rn_MJ_m2_day=Rn_MJ_day,
            T_C=clim["Tavg_C"],
            alpha=1.26
        ) * 365
        
    elif accuracy == "balanced":
        # Monthly aggregation (12 calculations)
        monthly_evap = []
        for month in range(1, 13):
            month_data = _get_monthly_climate_data(ts, month)
            E = de_bruin_keijman_daily(
                Rs_MJ_m2_day=month_data["Rs"] * 3.6 / month_data["days"],
                T_C=month_data["Tavg"],
                u2_m_s=month_data["wind"],
                RH=month_data["RH"] * 100
            )
            monthly_evap.append(E * month_data["days"])
        E_mm = sum(monthly_evap)
        
    else:  # "accurate"
        # Daily FAO Penman-Monteith (365 calculations)
        daily_evap = []
        for day in range(1, 367):
            day_data = _get_daily_climate_data(ts, day)
            # Would need full implementation of daily data extraction
            # This is a placeholder - actual implementation would loop through days
            pass
        E_mm = sum(daily_evap) if daily_evap else 0

    # Apply shading and wind block effects
    # Shading reduces radiation proportionally
    E_shaded_mm = E_mm * (1 - panel_coverage * 0.7)  # 70% of radiation blocked
    
    # Wind block affects aerodynamic component
    # In Penman-type equations, wind affects the second term
    wind_reduction = 1 - (1 - wind_block_factor) * 0.5  # 50% of wind effect
    
    E_final_mm = E_shaded_mm * wind_reduction
    
    # Convert to liters
    evap_liters = E_final_mm * water_area_m2
    
    return {
        "evap_mm_yr": E_final_mm,
        "evap_liters_yr": evap_liters
    }

# ============================================================
# Rainwater collection (Solar Waves)
# ============================================================

def get_openmeteo_daily_precipitation_sum_mm(*, latitude: float, longitude: float, year: int) -> List[float]:
    """Fetch daily precipitation_sum (mm/day) for a full calendar year from Open-Meteo archive/forecast.

    Returns a list of daily precipitation sums in mm. Length is typically 365 or 366.

    Notes:
      - Uses the archive endpoint first; falls back to forecast if needed.
      - Cached in-process by (rounded lat, rounded lon, year).
    """
    key = (_round_coord(latitude), _round_coord(longitude), int(year))
    if key in _PRECIP_CACHE:
        return _PRECIP_CACHE[key]

    start_date = f"{int(year)}-01-01"
    end_date = f"{int(year)}-12-31"

    url_archive = "https://archive-api.open-meteo.com/v1/archive"
    url_forecast = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }

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

        if url == url_forecast and r.status_code == 400:
            continue

    if data is None:
        if last_exc is not None:
            raise OpenMeteoError(f"Open-Meteo unreachable: {type(last_exc).__name__}: {last_exc}")
        raise OpenMeteoError(f"Open-Meteo error {last_status}: {last_text}")

    daily = data.get("daily") or {}
    precip = daily.get("precipitation_sum") or []
    out: List[float] = [float(x or 0.0) for x in precip]

    _PRECIP_CACHE[key] = out
    return out



def annual_rainwater_collection(
    *,
    latitude: float,
    longitude: float,
    year: int,
    total_panels: int,
    panel_width_m: float,
    panel_height_m: float,
    tilt_deg: float,
    eta: float = 0.80,
    coverage: float = 1.00,
) -> Dict[str, float]:
    """Estimate annual rainwater collection for Solar Waves (m^3 and liters).

    Model (V1):
      AnnualCollected_m3 = (AnnualRainfall_mm/1000) * A_catch_m2 * eta * coverage

      A_slope_m2 = total_panels * (panel_width_m * panel_height_m)
      A_catch_m2 = A_slope_m2 * cos(tilt_rad)

    - Rainfall is assumed vertical; therefore use horizontal projection (plan catchment).
    - eta bundles runoff + gutter capture + losses.
    - coverage can optionally derate for gaps/edge effects.
    """
    if total_panels <= 0:
        return {
            "annual_rainfall_mm": 0.0,
            "a_slope_m2": 0.0,
            "a_catch_m2": 0.0,
            "eta": float(eta),
            "coverage": float(coverage),
            "water_collected_m3": 0.0,
            "water_collected_liters": 0.0,
        }

    precip_mm = get_openmeteo_daily_precipitation_sum_mm(latitude=latitude, longitude=longitude, year=year)
    annual_rainfall_mm = float(sum(precip_mm))

    a_panel = float(panel_width_m) * float(panel_height_m)
    a_slope = float(total_panels) * a_panel

    tilt_rad = math.radians(float(tilt_deg))
    a_catch = a_slope * math.cos(tilt_rad)
    if a_catch < 0.0:
        a_catch = 0.0

    water_m3 = (annual_rainfall_mm / 1000.0) * a_catch * float(eta) * float(coverage)
    water_liters = water_m3 * 1000.0

    return {
        "annual_rainfall_mm": annual_rainfall_mm,
        "a_slope_m2": a_slope,
        "a_catch_m2": a_catch,
        "eta": float(eta),
        "coverage": float(coverage),
        "water_collected_m3": float(water_m3),
        "water_collected_liters": float(water_liters),
    }
