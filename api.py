from __future__ import annotations

import tempfile
import zipfile
import json
import os
import time

import datetime as dt

from typing import Dict, List, Sequence, Tuple, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sim_core import pv_energy_from_tau_samples, validate_base_array_simulation, get_meteo_timeseries
from sim_core import debug_compare_openmeteo_azimuth_conventions
import requests
from fastapi import Query


app = FastAPI(title="Solar Output API", version="0.5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://solarsim-frontend.pages.dev",
        "https://tiger-py.github.io",
        "https://sw-web-eight.vercel.app",
        "https://solarwaves.com.au",
        "https://www.solarwaves.com.au",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {
    "User-Agent": "SolarWaves/1.0 (https://solarwaves.com.au; nina@solarwaves.com.au)"
}

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"

# --- Simulation core ---
try:
    from sim_core import (
        annual_climate_means,
        annual_rainwater_collection,
        infer_wind_block_factor,
        simulate_annual_output,
        water_metrics_from_shading_samples,
        water_evaporation_lpy_from_kwh_m2,
        calculate_water_evaporation_optimized,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    simulate_annual_output = None  # type: ignore
    get_meteo_timeseries = None  # type: ignore
    infer_wind_block_factor = None  # type: ignore
    validate_base_array_simulation = None  # type: ignore
    water_evaporation_lpy_from_kwh_m2 = None  # type: ignore
    get_meteo_timeseries = None  # type: ignore
    annual_climate_means = None  # type: ignore
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

    # Copy multiplier for scaling
    copy_multiplier: Optional[int] = Field(default=1, ge=1)

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
    """Forward geocode using Open-Meteo API."""
    params = {"name": q, "count": count, "language": "en", "format": "json"}
    try:
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
    except Exception as e:
        # Log the error but return empty results gracefully
        print(f"Geocode error: {e}")
        return {"provider": "open-meteo", "results": [], "error": str(e)}

def _nominatim_reverse(lat: float, lon: float) -> Dict[str, Any]:
    """Reverse geocode using Nominatim."""
    import time
    
    url = f"{NOMINATIM_BASE}/reverse"
    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 10,
        "addressdetails": 1
    }
    
    # Retry logic with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(attempt * 1)  # Progressive delay: 0s, 1s, 2s
            
            r = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=20
            )
            r.raise_for_status()
            data = r.json()
            return {
                "provider": "nominatim",
                "display_name": data.get("display_name", f"Lat {lat}, Lon {lon}"),
                "lat": float(data.get("lat", lat)),
                "lon": float(data.get("lon", lon)),
                "address": data.get("address", {}),
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403 and attempt < max_retries - 1:
                print(f"Rate limited, retrying... (attempt {attempt + 1})")
                continue
            print(f"Reverse geocode HTTP error: {e}")
            # Fall through to fallback if last attempt
            if attempt == max_retries - 1:
                return {
                    "provider": "fallback",
                    "display_name": f"Lat {lat}, Lon {lon}",
                    "lat": float(lat),
                    "lon": float(lon),
                    "error": str(e),
                }
            
        except Exception as e:
            print(f"Reverse geocode error: {e}")
            if attempt < max_retries - 1:
                continue
            # Last attempt failed, return fallback
            return {
                "provider": "fallback",
                "display_name": f"Lat {lat}, Lon {lon}",
                "lat": float(lat),
                "lon": float(lon),
                "error": str(e),
            }
    
    # If we get here, all retries failed without a specific exception
    return {
        "provider": "fallback",
        "display_name": f"Lat {lat}, Lon {lon}",
        "lat": float(lat),
        "lon": float(lon),
        "error": "All retries failed",
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
     # Log the error but return empty results gracefully
        return {"provider": "open-meteo", "results": [], "error": str(e)}
 

@app.get("/reverse_geocode")
def reverse_geocode_endpoint(lat: float = Query(...), lon: float = Query(...)) -> Dict[str, Any]:
    return _nominatim_reverse(lat, lon)


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


@app.post("/validate_base_array")
def validate_base_array(req: SimRequest) -> Dict[str, Any]:
    """Validate base array configuration for potential simulation flaws."""
    _ensure_sim_core()
    
    buckets = _compute_orientation_buckets(req)
    
    diagnostics = validate_base_array_simulation(
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
    )
    
    return diagnostics


@app.post("/export_shaded_run")
def export_shaded_run(req: SimShadedRequest) -> FileResponse:
    """Export complete simulation data as ZIP archive for audit."""
    _ensure_sim_core()
    
    # Run the same simulation as /simulate_shaded
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
        accuracy="balanced"  # Use balanced accuracy by default
    )
    
    year_int = int(req.year) if req.year is not None else 2024
    
    # Get raw meteo data
    meteo = get_meteo_timeseries(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year_int
    )
    
    # Prepare export data
    export_data = {
        "simulation": {
            "base": base,
            "request": req.model_dump(),
            "buckets": buckets,
        },
        "water_metrics": {},
        "pv_metrics": {},
        "meteo_summary": {
            "times_utc": [t.isoformat() for t in meteo.times_utc[:24]],  # First day only
            "sample_ghi": meteo.ghi_w_m2[:24],
            "sample_dni": meteo.dni_w_m2[:24],
            "sample_dhi": meteo.dhi_w_m2[:24],
            "tair_c_mean": sum(meteo.tair_c) / len(meteo.tair_c) if meteo.tair_c else 0,
        }
    }
    
    # PV shading if present
    if req.pv_shading_samples:
        pv_samples = []
        for s in (req.pv_shading_samples or []):
            d = s.model_dump()
            d["tau"] = max(0.0, min(1.0, float(s.pv_factor())))
            pv_samples.append(d)
            
        pv_shaded = pv_energy_from_tau_samples(
            base=base,
            pv_samples=pv_samples,
        )
        export_data["pv_metrics"] = pv_shaded
    
    # Water metrics
    if req.shading_samples:
        wm = water_metrics_from_shading_samples(
            latitude=float(req.lat),
            longitude=float(req.lon),
            year=year_int,
            samples=[s.model_dump() for s in (req.shading_samples or [])],
            svf=float(req.svf),
        )
        export_data["water_metrics"] = wm
    
    # Create temporary ZIP file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Write main JSON data
            zf.writestr('simulation_data.json', json.dumps(export_data, indent=2, default=str))
            
            # Write sample files for each orientation
            for i, az in enumerate(buckets["surface_azimuths_deg"]):
                ts = get_tilted_timeseries(
                    latitude=float(req.lat),
                    longitude=float(req.lon),
                    tilt_deg=float(req.tilt_deg),
                    azimuth_deg=az,
                    year=year_int,
                )
                csv_data = "time_utc,poa_w_m2\n"
                for t, poa in zip(ts.times, ts.poa_w_m2):
                    csv_data += f"{t.isoformat()},{poa:.2f}\n"
                zf.writestr(f'poa_timeseries_az_{int(az)}.csv', csv_data)
   
    # Return file and clean up after download
    filename = f"sw_export_{req.lat:.2f}_{req.lon:.2f}_{year_int}.zip"
    response = FileResponse(
        tmp.name,
        media_type='application/zip',
        filename=filename
    )
    response.background = lambda: os.unlink(tmp.name)
    return response

@app.post("/simulate_snapshot_day")
def simulate_snapshot_day(req: SimRequest) -> Dict[str, Any]:

    _ensure_sim_core()

    buckets = _compute_orientation_buckets(req)
    print("[AZ BUCKETS]", buckets)

    # Run the same annual model and reuse mean POA as a proxy for daily mean POA.
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


    for az_deg, n in zip(buckets["surface_azimuths_deg"], buckets["panels_per_azimuth"]):
        orientations.append(
            {
                "orientation_key": _key(float(az_deg), tilt),
                "panels": int(n),
                "daily_mean_poa_w_m2": float(mean_poa) if mean_poa else 0.0,
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
    copy_multiplier = int(req.copy_multiplier) if hasattr(req, 'copy_multiplier') else 1
    if year_int is None:
        year_int = 2024

    # Calculate panel coverage (0-1)
    panel_area_total = float(req.total_panels) * float(req.panel_width_m) * float(req.panel_height_m)
    water_area = float(req.water_area_m2) if req.water_area_m2 is not None else 0.0
    panel_coverage = min(1.0, panel_area_total / max(1.0, water_area)) if water_area > 0 else 0.0
    
    wind_factor_covered = infer_wind_block_factor(array_type=req.array_type, height_m=req.height_m)

    wm = water_metrics_from_shading_samples(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year_int,
        samples=[s.model_dump() for s in (req.shading_samples or [])],
        svf=svf,
        elevation_m=float(req.height_m) if req.height_m is not None else 0.6,
        water_area_m2=water_area,
        panel_coverage=panel_coverage,
        wind_block_factor=wind_factor_covered,
        accuracy="balanced"
    )

    # Use the new optimized water evaporation calculator
    evap_results = calculate_water_evaporation_optimized(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year_int,
        water_area_m2=water_area * copy_multiplier,
        panel_coverage=panel_coverage,
        wind_block_factor=wind_factor_covered,
        accuracy="balanced"
    )

    # Extract evaporation results
    evap_mm_yr = evap_results.get("evap_mm_yr", 0)
    evap_liters_yr = evap_results.get("evap_liters_yr", 0)
    
    # For backward compatibility, calculate uncovered evaporation
    # (this is a simplified version - you may want to enhance this)
    evap_uncovered_lpy = calculate_water_evaporation_optimized(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year_int,
        water_area_m2=water_area * copy_multiplier,
        panel_coverage=0,  # No panels = no coverage
        wind_block_factor=1.0,  # Fully exposed
        accuracy="balanced"
    ).get("evap_liters_yr", 0)
    
    water_saved_lpy = max(0, evap_uncovered_lpy - evap_liters_yr)
    water_saved_pct = (water_saved_lpy / evap_uncovered_lpy * 100) if evap_uncovered_lpy > 0 else 0


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
        "water_saved_lpy": float(water_saved_lpy),
        "water_saved_pct": float(water_saved_pct),
        "water_area_m2": float(water_area * copy_multiplier),
        "evap_uncovered_lpy": float(evap_uncovered_lpy),
        "evap_covered_lpy": float(evap_liters_yr),
        "water_saved_lpy": float(water_saved_lpy),
        "water_saved_pct": float(water_saved_pct),
        "evap_model": {
            "use_penman": True,
            "K": 0.15,
            "lambda_MJ_per_kg": 2.45,
            "Tavg_C": 15.0,  # You may want to fetch these from climate data
            "RH_dec": 0.5,
            "Wavg_m_s": 3.0,
            "wind_block_factor_uncovered": 1.0,
            "wind_block_factor_covered": float(wind_factor_covered),
            "R_uncovered_MJ_m2_yr": 0.0,  # Calculate if needed
            "R_covered_MJ_m2_yr": 0.0,
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
