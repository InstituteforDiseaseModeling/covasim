
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.prepared import prep
import geopandas as gpd

__all__ = ['sewershed_id_map']


def _integer_sids(sids):
    """
    Turn the 'sid' property into integer IDs, which is what people.sewershed stores.

    If every sid is already an integer (1, 2, ... or "1", "2", ...) those values are used as they are.
    Otherwise (e.g. "SWR-01", "plant_A") the sewersheds are numbered 1, 2, 3, ... in file order.
    0 (default catchment) and -1 (outside all polygons) are reserved, so IDs must be >= 1.
    """
    as_str = sids.astype(str).str.strip()
    is_integer = as_str.str.fullmatch(r"-?\d+").all()
    if not is_integer and pd.api.types.is_float_dtype(sids): # e.g. 1.0, 2.0 from a numeric GeoJSON property
        is_integer = bool((sids % 1 == 0).all())
        as_str = sids.astype(int).astype(str)
    if is_integer:
        ids = as_str.astype(int).to_numpy()
        if (ids < 1).any():
            raise ValueError("Integer sewershed sid values must be >= 1 (0 and -1 are reserved for the default catchment and for outside all polygons).")
    else:
        ids = np.arange(1, len(sids) + 1)
    if len(np.unique(ids)) != len(ids): # Cannot normally happen, since the original sids are unique
        raise ValueError("Sewershed sid values must be unique.")
    return ids


def sewershed_id_map(filename):
    """
    Return {integer ID: original sid} for a sewershed file, i.e. the number stored in
    people.sewershed (and passed as sid= to WastewaterSampler) for each sewershed.
    """
    gdf = load_sewersheds(filename)
    return dict(zip(gdf["sid_num"].tolist(), gdf["sid"].tolist()))


def load_sewersheds(filename):
    """Load sewershed polygons from a GeoJSON/shapefile."""

    gdf = gpd.read_file(filename)

    if gdf.empty:
        raise ValueError("Sewershed file contains no polygons.")

    if "sid" not in gdf.columns:
        raise ValueError(
            "Sewershed GeoJSON must contain a 'sid' property."
        )

    if gdf["sid"].isna().any():
        raise ValueError("Every sewershed must have a sid.")

    if gdf["sid"].duplicated().any():
        raise ValueError("Sewershed sid values must be unique.")

    if not gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    ).all():
        raise ValueError(
            "Sewershed geometries must be Polygon or MultiPolygon."
        )

    if not gdf.geometry.is_valid.all():
        raise ValueError(
            "Sewershed GeoJSON contains invalid geometries."
        )

    gdf["sid_num"] = _integer_sids(gdf["sid"]) # Integer ID used in people.sewershed

    return gdf

def assign_sewersheds(people, pars):
    """Set people.sewershed (and people.x/y if people_coords is given); returns {integer ID: original sid}."""
    coords = pars.get("people_coords")

    if coords is not None:
        coords = np.asarray(coords, dtype=float)

        if coords.shape != (len(people), 2):
            raise ValueError(
                "people_coords must have shape (pop_size, 2)"
            )

        if not np.isfinite(coords).all():
            raise ValueError(
                "people_coords must contain finite values"
            )

        people.x[:] = coords[:, 0]
        people.y[:] = coords[:, 1]

    filename = pars.get("sewershed_file")

    if filename is None:
        people.sewershed[:] = 0
        return {}
        
    if not np.isfinite(people.x).all() or not np.isfinite(people.y).all():
        raise ValueError(
            "Sewershed assignment requires valid coordinates."
        )

    sewersheds = load_sewersheds(filename)

    # 1. Ensure CRS is explicitly set to EPSG:4326 if missing
    if sewersheds.crs is None:
        sewersheds = sewersheds.set_crs("EPSG:4326")

    # 2. Integer IDs ('sid_num') were made when the file was loaded

    # 3. Create agent Point geometries (x = lon, y = lat)
    points = gpd.GeoDataFrame(
        {"agent_id": np.arange(len(people))},
        geometry=[Point(x, y) for x, y in zip(people.x, people.y)],
        crs=sewersheds.crs,
    )

    # 4. Perform spatial join (using 'intersects' or 'within')
    joined = gpd.sjoin(
        points,
        sewersheds[["sid_num", "geometry"]],
        how="left",
        predicate="intersects",
    )

    # 5. Drop potential duplicate matches if an agent touches overlapping polygons
    joined = joined.drop_duplicates(subset=["agent_id"], keep="first")

    # 6. Map integer IDs back to the array
    assignments = np.full(len(people), -1, dtype=int)
    
    # Fill matched IDs using agent_id as the array index
    matched_agents = joined["agent_id"].to_numpy()
    matched_sids = joined["sid_num"].fillna(-1).to_numpy(dtype=int)
    
    assignments[matched_agents] = matched_sids

    people.sewershed[:] = assignments

    return dict(zip(sewersheds["sid_num"].tolist(), sewersheds["sid"].tolist())) # {integer ID: original sid}
