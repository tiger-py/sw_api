from __future__ import annotations

import datetime as dt

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sim_core import pv_energy_from_tau_samples
import requests
from sim_core import debug_compare_openmeteo_azimuth_conventions
from fastapi import Query


app = FastAPI(title="Solar Output API", version="0.5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://solarsim-frontend.pages.dev",
        "https://tiger-py.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {"User-Agent": "SolarWaves/1.0 (contact@example.com)"}

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"

# --- Simulation core ---
# sim_core.py is expected to be present in the same service.
try:
    from sim_core import (
        simulate_annual_output,
        water_metrics_from_shading_samples,
        get_meteo_timeseries,
        annual_climate_means,
        infer_wind_block_factor,
        water_evaporation_lpy_from_kwh_m2,
        annual_rainwater_collection,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    simulate_annual_output = None  # type: ignore
    water_metrics_from_shading_samples = None  # type: ignore
    get_meteo_timeseries = None  # type: ignore
    annual_climate_means = None  # type: ignore
    infer_wind_block_factor = None  # type: ignore
    water_evaporation_lpy_from_kwh_m2 = None  # type: ignore
    annual_rainwater_collection = None  # type: ignore
    _SIM_IMPORT_ERROR = str(e)


class SimRequest(BaseModel):
    # Location
    lat: float
    lon: float

    # Array definition
    array_type: str = Field(default="waves")  # "waves" or "roof"
    azimuth_deg: float = Field(default=180.0)
    tilt_deg: float = Field(default=12.5)
    total_panels: int = Field(default=10, ge=1)

    # Panel geometry
    panel_width_m: float = Field(default=1.0, gt=0)
    panel_height_m: float = Field(default=1.6, gt=0)

    # Electrical / thermal parameters (optional; sim_core has defaults)
    eff_stc: Optional[float] = Field(default=None, ge=0, le=0.35)
    noct: Optional[float] = Field(default=None)
    temp_coeff: Optional[float] = Field(default=None)  # e.g. -0.0035
    cooling_offset: Optional[float] = Field(default=None)

    # Water surface (only relevant for water mode)
    water_width_m: Optional[float] = Field(default=None)

    # Water evaporation model inputs (optional; used for Ladybug-aligned water savings)
    height_m: Optional[float] = Field(default=None, description="Height of array above water surface (m)")
    water_area_m2: Optional[float] = Field(default=None, description="Water surface area considered (m^2)")
    use_penman: Optional[bool] = Field(default=True, description="Use modified Penman-style radiation model")
    K_evap: Optional[float] = Field(default=None, description="Empirical evaporation coefficient K (e.g., 0.15 in report)")
    lambda_evap_mj_per_kg: Optional[float] = Field(default=None, description="Latent heat lambda in MJ/kg (default 2.45)")
    wind_block_factor: Optional[float] = Field(default=None, description="Wind blocking factor (0..1); 1 = open water")


class ShadingSample(BaseModel):
    time_utc: str

    # Backward-compatible (old frontend)
    tau: float = Field(default=1.0, ge=0.0, le=1.0)

    # New frontend fields (optional)
    tau_shadow: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    f_beam: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    weight_hours: float = Field(gt=0.0)

    def pv_factor(self) -> float:
        if self.f_beam is not None:
             return float(self.f_beam)
        if self.tau_shadow is not None:
             return float(self.tau_shadow)
        if self.tau is not None:
            return float(self.tau)
        return 1.0



class SimShadedRequest(SimRequest):
    year: Optional[int] = None
    svf: float = Field(ge=0.0, le=1.0)
    svf_n_rays: Optional[int] = None
    svf_scheme: Optional[str] = None
    shading_samples: List[ShadingSample] = Field(default_factory=list)
    pv_shading_samples: List[ShadingSample] = Field(default_factory=list)


def _open_meteo_geocode(q: str, count: int = 5) -> Dict[str, Any]:
    params = {"name": q, "count": count, "language": "en", "format": "json"}
    r = requests.get(OPEN_METEO_GEOCODE, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("results", [])[:count]:
        results.append(
            {
                "display_name": ", ".join(
                    [x for x in [item.get("name"), item.get("admin1"), item.get("country")] if x]
                ),
                "lat": float(item["latitude"]),
                "lon": float(item["longitude"]),
            }
        )
    return {"provider": "open-meteo", "results": results}


def _nominatim_reverse(lat: float, lon: float) -> Dict[str, Any]:
    url = f"{NOMINATIM_BASE}/reverse"
    params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 10, "addressdetails": 1}
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    return {
        "provider": "nominatim",
        "display_name": data.get("display_name", f"Lat {lat}, Lon {lon}"),
        "lat": float(data.get("lat", lat)),
        "lon": float(data.get("lon", lon)),
        "address": data.get("address", {}),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/geocode")
def geocode(q: str = Query(..., min_length=1), count: int = Query(5, ge=1, le=10)) -> Dict[str, Any]:
    """Forward geocode: place name -> candidate coordinates."""
    try:
        return _open_meteo_geocode(q, count=count)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"geocode_failed: {e}")


@app.get("/reverse_geocode")
def reverse_geocode(lat: float = Query(...), lon: float = Query(...)) -> Dict[str, Any]:
    """Reverse geocode: coordinates -> display name."""
    try:
        return _nominatim_reverse(lat, lon)
    except Exception as e:
        # Fail soft: still return the lat/lon so the frontend can proceed.
        return {
            "provider": "fallback",
            "display_name": f"Lat {lat}, Lon {lon}",
            "lat": float(lat),
            "lon": float(lon),
            "error": str(e),
        }


def _ensure_sim_core() -> None:
    if (
        simulate_annual_output is None
        or water_metrics_from_shading_samples is None
        or get_meteo_timeseries is None
        or annual_climate_means is None
        or infer_wind_block_factor is None
        or water_evaporation_lpy_from_kwh_m2 is None
    ):
        raise HTTPException(
            status_code=500,
            detail=f"Simulation core not available. Import error: {_SIM_IMPORT_ERROR}",
        )


def _compute_panel_area_m2(req: SimRequest) -> float:
    return float(req.panel_width_m) * float(req.panel_height_m)


def _compute_orientation_buckets(req: SimRequest) -> Dict[str, Any]:
    """Map frontend array_type to (surface_azimuths_deg, panels_per_azimuth).

    Conventions:
      - Frontend/UI azimuth: 0=N, 90=E, 180=S, 270=W.

    Buckets:
      - Roof/canopy: single orientation at req.azimuth_deg
      - Waves: two opposing faces (az and az+180). This matches a V/accordion
        cross-section better than +/-90 (perpendicular faces).
    """
    at = (req.array_type or "waves").strip().lower()

    def wrap360(deg: float) -> float:
        d = float(deg) % 360.0
        return d if d >= 0 else d + 360.0

    az = wrap360(req.azimuth_deg)

    if at in ("roof", "solar_roof", "canopy", "solar_canopy"):
        return {"surface_azimuths_deg": [az], "panels_per_azimuth": [int(req.total_panels)]}

    # Default: "waves"
    # Split evenly into two opposing faces. If odd, bias to the first bucket.
    n = int(req.total_panels)
    n1 = (n + 1) // 2
    n2 = n - n1

    a0 = az
    a1 = az + 180.0
    return {"surface_azimuths_deg": [a0, a1], "panels_per_azimuth": [n1, n2]}

@app.get("/debug/openmeteo_az")
def debug_openmeteo_az(
    lat: float = Query(...),
    lon: float = Query(...),
    tilt: float = Query(25.0),
    year: int = Query(2025),
):
    return debug_compare_openmeteo_azimuth_conventions(
        latitude=lat,
        longitude=lon,
        tilt_deg=tilt,
        year=year,
    )

@app.post("/simulate")
def simulate(req: SimRequest) -> Dict[str, Any]:
    """Unshaded annual simulation."""
    _ensure_sim_core()

    buckets = _compute_orientation_buckets(req)
    print("[AZ BUCKETS]", buckets)

    out = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=float(req.tilt_deg),
        surface_azimuths_deg=buckets["surface_azimuths_deg"],
        panels_per_azimuth=buckets["panels_per_azimuth"],
        panel_area_m2=_compute_panel_area_m2(req),
        eff_stc=float(req.eff_stc) if req.eff_stc is not None else 0.20,
        noct=float(req.noct) if req.noct is not None else 45.0,
        temp_coeff=float(req.temp_coeff) if req.temp_coeff is not None else -0.0035,
        cooling_offset=float(req.cooling_offset) if req.cooling_offset is not None else 0.0,
        year=None,
    )

    # The frontend expects these keys.
    return {
        "annual_energy_kwh": out.get("annual_energy_kwh"),
        "mean_poa_w_m2": out.get("mean_poa_w_m2"),
        "details": out,
    }



@app.post("/simulate_snapshot_day")
def simulate_snapshot_day(req: SimRequest) -> Dict[str, Any]:
    """Lightweight snapshot endpoint used by the frontend map picker/UI.

    The current frontend uses this to color panels for a representative day.
    This implementation returns a conservative, deterministic payload that matches
    the keys the frontend expects, without requiring additional solar-position
    dependencies.

    If you later want true day-specific outputs, this is the place to add them.
    """
    _ensure_sim_core()

    buckets = _compute_orientation_buckets(req)
    print("[AZ BUCKETS]", buckets)

    # Run the same annual model and reuse mean POA as a proxy for daily mean POA.
    # Keep arguments aligned with SimRequest fields to avoid runtime AttributeErrors.
    # simulate_annual_output in your sim_core takes these parameters.
    out = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=float(req.tilt_deg),
        surface_azimuths_deg=buckets["surface_azimuths_deg"],
        panels_per_azimuth=buckets["panels_per_azimuth"],
        panel_area_m2=_compute_panel_area_m2(req),
        eff_stc=float(req.eff_stc) if req.eff_stc is not None else 0.20,
        noct=float(req.noct) if req.noct is not None else 45.0,
        temp_coeff=float(req.temp_coeff) if req.temp_coeff is not None else -0.0035,
        cooling_offset=float(req.cooling_offset) if req.cooling_offset is not None else 0.0,
        year=None,
    )

    mean_poa = out.get("mean_poa_w_m2")
    tilt = float(req.tilt_deg)

    def _key(az_deg: float, tilt_deg: float) -> str:
        az = az_deg % 360.0
        az_r = round(az, 1)
        t_r = round(float(tilt_deg), 1)
        return f"{az_r:.1f}|{t_r:.1f}"

    orientations = []

    print("[AZ INPUT]", {
        "ui_azimuth_deg": float(req.azimuth_deg),
        "surface_azimuths_deg": list(buckets["surface_azimuths_deg"]),
        "panels_per_azimuth": list(buckets["panels_per_azimuth"]),
        "tilt_deg": float(req.tilt_deg),
        "array_type": req.array_type,
    })


    for az_deg, n in zip(buckets["surface_azimuths_deg"], buckets["panels_per_azimuth"]):
        orientations.append(
            {
                "orientation_key": _key(float(az_deg), tilt),
                "panels": int(n),
                "daily_mean_poa_w_m2": mean_poa,
            }
        )

    # Provide deterministic "snapshot sun" angles so the frontend can orient shadows.
    return {
        "snapshot_sun_az_deg": float(req.azimuth_deg) % 360.0,
        "snapshot_sun_el_deg": 45.0,
        "orientations": orientations,
        "details": out,
    }



@app.post("/simulate_shaded")
def simulate_shaded(req: SimShadedRequest) -> Dict[str, Any]:
    """Shaded annual simulation (water mode): applies an effective shading factor."""
    _ensure_sim_core()

    buckets = _compute_orientation_buckets(req)

    base = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=float(req.tilt_deg),
        surface_azimuths_deg=buckets["surface_azimuths_deg"],
        panels_per_azimuth=buckets["panels_per_azimuth"],
        panel_area_m2=_compute_panel_area_m2(req),
        eff_stc=float(req.eff_stc) if req.eff_stc is not None else 0.20,
        noct=float(req.noct) if req.noct is not None else 45.0,
        temp_coeff=float(req.temp_coeff) if req.temp_coeff is not None else -0.0035,
        cooling_offset=float(req.cooling_offset) if req.cooling_offset is not None else 0.0,
        year=int(req.year) if req.year is not None else None,
    )

    annual = float(base.get("annual_energy_kwh") or 0.0)
    mean_poa = float(base.get("mean_poa_w_m2") or 0.0)

    pv_shaded: Dict[str, Any] = {}
    tau_pv_mean = 1.0

    if req.pv_shading_samples:
        # Normalize samples for sim_core: always provide "tau" as the factor to apply.
        pv_samples = []
        for s in (req.pv_shading_samples or []):
            d = s.model_dump()
            d["tau"] = max(0.0, min(1.0, float(s.pv_factor())))
            pv_samples.append(d)

        pv_shaded = pv_energy_from_tau_samples(
            base=base,
            pv_samples=pv_samples,
        )
        tau_pv_mean = float((pv_shaded or {}).get("tau_pv_mean", 1.0) or 1.0)


    # Compute weighted mean transmittance tau from samples.
    # If no samples, fall back to tau=1.0 (no additional shading).
    if req.shading_samples:
        wsum = 0.0
        tsum = 0.0
        for s in req.shading_samples:
            w = float(s.weight_hours)
            t = float(s.tau)
            wsum += w
            tsum += w * t
        tau_water_mean = (tsum / wsum) if wsum > 0 else 1.0
    else:
        tau_water_mean = 1.0

    # Effective factor: combine sky-view factor and beam transmittance proxy.
    # This is a simplified model; you can refine later.
    svf = float(req.svf)
    effective = max(0.0, min(1.0, svf * tau_water_mean))
    
    # Water-plane metrics expected by the frontend
    year_int = int(req.year) if req.year is not None else None
    # Default to 2024 if year not provided (last full year), keeps results deterministic.
    if year_int is None:
        year_int = 2024

    wm = water_metrics_from_shading_samples(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year_int,
        samples=[s.model_dump() for s in (req.shading_samples or [])],
        svf=svf,
    )

    # --- Ladybug-aligned evaporation model (liters/year) ---
    water_area_m2 = float(req.water_area_m2) if req.water_area_m2 is not None else 0.0
    # If not provided, we cannot compute absolute liters/year; percentages still returned.
    if water_area_m2 < 0:
        water_area_m2 = 0.0

    ts_w = get_meteo_timeseries(latitude=float(req.lat), longitude=float(req.lon), year=year_int)
    clim = annual_climate_means(ts_w)
    Tavg_C = float(clim.get("Tavg_C", 0.0))
    RH_dec = float(clim.get("RH_dec", 0.0))
    Wavg_m_s = float(clim.get("Wavg_m_s", 0.0))

    K_evap = float(req.K_evap) if req.K_evap is not None else 0.15
    lam_evap = float(req.lambda_evap_mj_per_kg) if req.lambda_evap_mj_per_kg is not None else 2.45
    use_penman = bool(req.use_penman) if req.use_penman is not None else True

    # Wind exposure factors (1 = fully exposed open water)
    wind_factor_uncovered = 1.0
    wind_factor_covered = (
        float(req.wind_block_factor)
        if req.wind_block_factor is not None
        else infer_wind_block_factor(array_type=req.array_type, height_m=req.height_m)
    )

    evap_uncovered = water_evaporation_lpy_from_kwh_m2(
        water_kwh_m2_yr=float(wm.get("water_baseline_kwh_m2", 0.0)),
        water_area_m2=water_area_m2,
        Tavg_C=Tavg_C,
        RH_dec=RH_dec,
        Wavg_m_s=Wavg_m_s,
        K=K_evap,
        lambda_MJ_per_kg=lam_evap,
        use_penman=use_penman,
        wind_block_factor=wind_factor_uncovered,
    )
    evap_covered = water_evaporation_lpy_from_kwh_m2(
        water_kwh_m2_yr=float(wm.get("water_shaded_kwh_m2", 0.0)),
        water_area_m2=water_area_m2,
        Tavg_C=Tavg_C,
        RH_dec=RH_dec,
        Wavg_m_s=Wavg_m_s,
        K=K_evap,
        lambda_MJ_per_kg=lam_evap,
        use_penman=use_penman,
        wind_block_factor=wind_factor_covered,
    )

    evap_uncovered_lpy = float(evap_uncovered.get("evap_lpy_wind", 0.0))
    evap_covered_lpy = float(evap_covered.get("evap_lpy_wind", 0.0))
    water_saved_lpy = max(0.0, evap_uncovered_lpy - evap_covered_lpy)
    water_saved_pct = 0.0 if evap_uncovered_lpy <= 1e-9 else (100.0 * water_saved_lpy / evap_uncovered_lpy)

    ann_shaded = (pv_shaded or {}).get("annual_energy_kwh_shaded_pv", annual)
    if ann_shaded is None:
        ann_shaded = annual

    tau_pv_mean = (pv_shaded or {}).get("tau_pv_mean", 1.0)
    if tau_pv_mean is None:
        tau_pv_mean = 1.0


    return {
        # PV energy/POA should come from the PV model (base) and should not be scaled by water SVF/tau.
        "annual_energy_kwh": annual,
        "mean_poa_w_m2": mean_poa,
        "annual_energy_kwh_shaded_pv": float(ann_shaded),
        "monthly_energy_kwh_shaded_pv": pv_shaded.get("monthly_energy_kwh_shaded_pv", base.get("monthly_energy_kwh", {})),
        "pv_shading": {
            "tau_pv_mean": float(tau_pv_mean),
            "tau_pv_monthly": pv_shaded.get("tau_pv_monthly", {}),
            "n_pv_samples": len(req.pv_shading_samples or []),
        },

        # Water metrics used for savings scaling in water mode
        "water_baseline_kwh_m2": float(wm.get("water_baseline_kwh_m2", 0.0)),
        "water_shaded_kwh_m2": float(wm.get("water_shaded_kwh_m2", 0.0)),
        "water_reduction_pct": float(wm.get("water_reduction_pct", 0.0)),

        # Ladybug-aligned evaporation outputs (liters/year)
        "water_area_m2": float(water_area_m2),
        "evap_uncovered_lpy": float(evap_uncovered_lpy),
        "evap_covered_lpy": float(evap_covered_lpy),
        "water_saved_lpy": float(water_saved_lpy),
        "water_saved_pct": float(water_saved_pct),
        "evap_model": {
            "use_penman": bool(use_penman),
            "K": float(K_evap),
            "lambda_MJ_per_kg": float(lam_evap),
            "Tavg_C": float(Tavg_C),
            "RH_dec": float(RH_dec),
            "Wavg_m_s": float(Wavg_m_s),
            "wind_block_factor_uncovered": float(wind_factor_uncovered),
            "wind_block_factor_covered": float(wind_factor_covered),
            "R_uncovered_MJ_m2_yr": float(evap_uncovered.get("R_MJ_m2_yr", 0.0)),
            "R_covered_MJ_m2_yr": float(evap_covered.get("R_MJ_m2_yr", 0.0)),
        },

        # Debug shading fields
        "shading": {
            "svf": svf,
            "tau_mean": float(tau_water_mean),
            "effective_factor": effective,
            "n_samples": len(req.shading_samples),
            "svf_n_rays": req.svf_n_rays,
            "svf_scheme": req.svf_scheme,
        },
        "details": base,
    }

@app.post("/rainwater_annual")
def rainwater_annual(req: SimRequest) -> Dict[str, Any]:
    """Annual rainwater collection estimate for Solar Waves (V1: last calendar year).

    This endpoint is intentionally separate from /simulate and /simulate_shaded so that
    existing simulation functions and payloads remain untouched.
    """
    _ensure_sim_core()

    if (req.array_type or "").lower() != "waves":
        return {
            "enabled": False,
            "reason": "rainwater metric is only implemented for array_type='waves'",
        }

    year = int(dt.datetime.utcnow().year) - 1

    out = annual_rainwater_collection(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year,
        total_panels=int(req.total_panels),
        panel_width_m=float(req.panel_width_m),
        panel_height_m=float(req.panel_height_m),
        tilt_deg=float(req.tilt_deg),
        eta=0.80,
        coverage=1.00,
    )

    return {
        "enabled": True,
        "year": year,
        **out,
    }
