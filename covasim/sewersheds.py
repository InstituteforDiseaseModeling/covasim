
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.prepared import prep
import geopandas as gpd


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

    return gdf

def assign_sewersheds(people, pars):
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
        return
        
    if not np.isfinite(people.x).all() or not np.isfinite(people.y).all():
        raise ValueError(
            "Sewershed assignment requires valid coordinates."
        )

    sewersheds = load_sewersheds(filename)

    # 1. Ensure CRS is explicitly set to EPSG:4326 if missing
    if sewersheds.crs is None:
        sewersheds = sewersheds.set_crs("EPSG:4326")

    # 2. Extract numeric integer IDs from string 'sid' (e.g., "SWR-01" -> 1)
    sewersheds["sid_num"] = pd.to_numeric(
        sewersheds["sid"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce"
    ).fillna(-1).astype(int)

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
