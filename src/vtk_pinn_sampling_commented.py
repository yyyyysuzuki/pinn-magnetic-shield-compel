
# -*- coding: utf-8 -*-
"""
VTK -> PINN sampling & visualization (fully commented)
------------------------------------------------------

This script reads a *Legacy ASCII* VTK unstructured mesh, extracts triangle cells,
derives material labels per triangle from CELL_DATA/SCALARS, and produces:

1) Area-proportional **interior samples** for PINN training.
   - Uses barycentric random sampling inside each triangle.
   - Enforces a lower bound "tau" on barycentric coordinates so that
     sampled points do **not fall on triangle edges** (which represent boundaries/interfaces).

2) **Boundary samples** on the outer boundary only:
   - The bottom (y ~ ymin) is tagged as Neumann (bc_type=1).
   - Left (x ~ xmin), right (x ~ xmax), and top (y ~ ymax) are tagged as Dirichlet (bc_type=0).
   - Each boundary sample is the midpoint of an outer boundary edge.
   - Outward unit normals are provided for potential Neumann loss usage.

3) A **preview PNG** overlaying:
   - Mesh edges
   - Interior sample points (colored by material label)
   - Boundary points (square markers for Dirichlet, triangles for Neumann)
   - Optional short arrows to show Neumann outward normals

4) A compressed **.npz** file with arrays ready for PINN training:
   X (interior points), mat_id, elem_id, X_bc, bc_type, n_bc, bbox, tau.

Assumptions:
- Input is *Legacy ASCII VTK*. For binary or XML (.vtu) use `meshio` instead.
- Materials are provided as a cell scalar array. The array named "Material" is
  preferred; otherwise the first available SCALARS array is used.
- Only triangle cells (VTK cell type 5) are used for sampling and visualization.

Author: ysuzuki
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------- Configuration --------------------

# Input/Output paths
VTK_PATH = Path("../data/gen_0ind_0_MaterialConfig.vtk")
OUT_NPZ  = Path("../data/pinn_points_from_vtk.npz")
OUT_PNG  = Path("../results/sampling_preview_from_vtk.png")

# Sampling parameters
POINTS_TOTAL: int = 20000   # Total number of interior samples to generate
TAU: float = 1e-3            # Barycentric coordinate floor to avoid edges
RANDOM_SEED: int = 42        # RNG seed for reproducibility

# -------------------- VTK Parser --------------------

def parse_vtk_legacy_ascii(path: Path) -> Tuple[np.ndarray, List[List[int]], np.ndarray, Dict[str, np.ndarray]]:
    """
    Parse a *Legacy ASCII* VTK file to extract:
      - POINTS: float coordinates (N,3)
      - CELLS: connectivity list of variable length (list of index lists)
      - CELL_TYPES: int codes (M,)
      - CELL_DATA/SCALARS: dict[name] -> (M,) float arrays

    Notes on format:
      POINTS <N> <dtype>
        <x y z> repeated N times
      CELLS <M> <K>
        <k_0 i0_0 ... i0_{k0-1}>  <k_1 ...>  ... length sums to K
      CELL_TYPES <M>
        <t0 t1 ... t_{M-1}>   where e.g. triangle = 5
      CELL_DATA <M>
        SCALARS <name> <dtype> [numComp]
        LOOKUP_TABLE <name>
        <v0 v1 ... v_{M-1}>
        (possibly multiple SCALARS blocks)

    Parameters
    ----------
    path : Path
        Path to the Legacy ASCII VTK file.

    Returns
    -------
    points : (N,3) float64
        Node coordinates.
    cells : list[list[int]]
        Unstructured connectivity per cell (node indices).
    cell_types : (M,) int32
        VTK cell type code per cell.
    cell_scalars : dict[str, (M,) float64]
        Scalar arrays attached at the CELL_DATA level, e.g. "Material".
        If missing, returns an empty dict.
    """
    # Tokenize the entire file (whitespace-separated).
    toks: List[str] = []
    with open(path, "r") as f:
        for ln in f.read().splitlines():
            toks.extend(ln.strip().split())

    # Quick sanity: ensure ASCII is mentioned in header block
    if "ASCII" not in toks:
        raise ValueError("This parser only supports Legacy ASCII VTK.")

    def find_token(token: str, start: int = 0) -> int:
        """Find index of a token (case-insensitive). Return -1 if not found."""
        T = token.upper()
        for i in range(start, len(toks)):
            if toks[i].upper() == T:
                return i
        return -1

    # ----- POINTS -----
    i = find_token("POINTS", 0)
    if i < 0:
        raise ValueError("POINTS block not found.")
    npts = int(float(toks[i+1]))
    # dtype token is toks[i+2], but we don't need it for ASCII parser
    p0 = i + 3
    pflat = list(map(float, toks[p0:p0 + 3*npts]))
    points = np.asarray(pflat, dtype=np.float64).reshape(npts, 3)

    # ----- CELLS -----
    j = find_token("CELLS", p0 + 3*npts)
    if j < 0:
        raise ValueError("CELLS block not found.")
    ncells = int(float(toks[j+1]))
    total_ints = int(float(toks[j+2]))  # sum of (1 + nodes_per_cell)
    clist = list(map(int, toks[j+3:j+3+total_ints]))

    cells: List[List[int]] = []
    idx = 0
    for _ in range(ncells):
        k = clist[idx]; idx += 1       # number of nodes in this cell
        ids = clist[idx:idx+k]; idx += k
        cells.append(ids)

    # ----- CELL_TYPES -----
    k = find_token("CELL_TYPES", j + 3 + total_ints)
    if k < 0:
        raise ValueError("CELL_TYPES block not found.")
    nct = int(float(toks[k+1]))
    ctyp = list(map(int, toks[k+2:k+2+nct]))
    cell_types = np.asarray(ctyp, dtype=np.int32)
    if nct != ncells:
        raise ValueError("CELL_TYPES count does not match CELLS count.")

    # ----- CELL_DATA / SCALARS (optional) -----
    cell_scalars: Dict[str, np.ndarray] = {}
    m = find_token("CELL_DATA", k + 2 + nct)
    if m >= 0:
        ncd = int(float(toks[m+1]))  # number of cells the data refers to
        pos = m + 2
        while True:
            s = find_token("SCALARS", pos)
            if s < 0:
                break  # no more SCALARS blocks
            name = toks[s+1]
            # dtype token at s+2 (unused for ASCII)
            p = s + 3
            # Optional numComponents:
            try:
                _maybe_comp = int(float(toks[p]))
                p += 1  # skip if numeric
            except Exception:
                pass
            # Then expect LOOKUP_TABLE <name>
            lt = find_token("LOOKUP_TABLE", p)
            if lt < 0:
                break
            # Values start after LOOKUP_TABLE <name>
            v0 = lt + 2
            vals = list(map(float, toks[v0:v0+ncd]))
            arr = np.asarray(vals, dtype=np.float64)
            cell_scalars[name] = arr
            pos = v0 + ncd  # continue in case of more SCALARS

    return points, cells, cell_types, cell_scalars


# -------------------- Geometry Utilities --------------------

def tri_areas(points2d: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """
    Compute triangle areas in 2D for an array of triangles.

    Parameters
    ----------
    points2d : (N,2) float
        2D node coordinates.
    tris : (M,3) int
        Triangle connectivity (node indices), each row [i, j, k].

    Returns
    -------
    areas : (M,) float
        Triangle areas (positive).
    """
    a = points2d[tris[:,0]]
    b = points2d[tris[:,1]]
    c = points2d[tris[:,2]]
    # 2D signed area formula: 0.5 * | (b-a) x (c-a) |
    return 0.5 * np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0]))


def build_edge_adjacency(tris: np.ndarray) -> Dict[Tuple[int,int], List[int]]:
    """
    Build an edge->adjacent-triangles map.

    Each triangle contributes three undirected edges; we store each edge
    with sorted (min,max) node order so that the same geometric edge maps
    to the same key.

    Parameters
    ----------
    tris : (M,3) int
        Triangle connectivity.

    Returns
    -------
    edge_to_elems : dict[(i,j)] -> list of triangle indices
        - Outer boundary edges: list length == 1
        - Interior edges: list length == 2
    """
    edge_to_elems: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    for eid, (i, j, k) in enumerate(tris):
        for e in (tuple(sorted((i, j))), tuple(sorted((j, k))), tuple(sorted((k, i)))):
            edge_to_elems[e].append(eid)
    return edge_to_elems


# -------------------- Sampling Routines --------------------

def sample_in_triangles(points2d: np.ndarray,
                        tris: np.ndarray,
                        mat_per_tri: np.ndarray,
                        n_total: int,
                        tau: float = 1e-6,
                        seed: int = 42,
                        material_weights:Optional[object] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Area-proportional interior sampling with **edge avoidance**.

    Strategy:
      - Determine each triangle's share of samples proportional to its area.
      - For each triangle, draw random (u,v) in [0,1]^2 and fold (u+v>1) back
        into the unit triangle. Then w=1-u-v.
      - Enforce min(u,v,w) >= tau so that points stay strictly inside and away
        from edges (which correspond to u=0, v=0, or w=0).

    Parameters
    ----------
    points2d : (N,2) float
        Node coordinates.
    tris : (M,3) int
        Triangle connectivity.
    mat_per_tri : (M,) int
        Material label for each triangle.
    n_total : int
        Total number of interior points to generate.
    tau : float
        Barycentric minimum threshold; choose small (e.g., 1e-3) to avoid edges.
    seed : int
        RNG seed.

    Returns
    -------
    X : (N,2) float32
        Interior sample points.
    mat_ids : (N,) int32
        Material label inherited from the triangle that produced each point.
    elem_ids : (N,) int64
        Triangle index (for debugging/visualization).
    """
    rng = np.random.default_rng(seed)

    # Allocate sample counts per triangle according to area
    areas = tri_areas(points2d, tris)
    probs = areas.copy()

    if material_weights is not None:
        if isinstance(material_weights, dict):
            # dict: {材料ID: 重み} を安全に適用。未指定IDは 1.0
            w_tri = np.array([float(material_weights.get(int(m), 1.0))
                              for m in mat_per_tri], dtype=np.float64)
        else:
            # 配列/リスト: インデックス=材料ID として参照
            arr = np.asarray(material_weights, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError("material_weights は 1次元配列か dict を渡してください。")
            if mat_per_tri.min() < 0:
                raise ValueError("材料IDに負が含まれています。配列参照は不可なので dict を使うか、IDを非負に正規化してください。")
            max_id = int(mat_per_tri.max())
            if max_id >= arr.shape[0]:
                raise ValueError(f"material_weights の長さ({arr.shape[0]})が材料IDの最大値({max_id})に足りません。")
            w_tri = arr[mat_per_tri]

        probs *= w_tri

    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("面積×重みの合計が不正です。面積や material_weights を確認してください。")

    probs /= s
    counts = np.floor(n_total * probs + 1e-9).astype(int)

    # Distribute any remaining samples to the largest triangles
    resid = n_total - counts.sum()
    if resid > 0:
        counts[np.argsort(-probs)[:resid]] += 1

    # Preload triangle corner coordinates for vectorization convenience
    A = points2d[tris[:,0]]
    B = points2d[tris[:,1]]
    C = points2d[tris[:,2]]

    P_all: List[np.ndarray] = []
    M_all: List[np.ndarray] = []
    E_all: List[np.ndarray] = []

    for eid, k in enumerate(counts):
        if k <= 0:
            continue

        a, b, c = A[eid], B[eid], C[eid]

        # Draw points in batches until we accumulate k good points
        need = k
        buf: List[np.ndarray] = []
        while need > 0:
            batch = max(need * 2, 1024)  # oversample to reduce while-iterations
            u = rng.random(batch)
            v = rng.random(batch)
            # Fold [0,1]^2 into the unit simplex (triangle):
            # if u+v > 1, map (u,v) to (1-u, 1-v)
            mask = (u + v > 1.0)
            u[mask] = 1.0 - u[mask]
            v[mask] = 1.0 - v[mask]
            w = 1.0 - u - v

            # Keep only points that are *strictly interior* by tau margin
            good = (u >= tau) & (v >= tau) & (w >= tau)
            u, v, w = u[good], v[good], w[good]

            # How many do we still need?
            take = min(need, u.size)
            if take > 0:
                u, v, w = u[:take], v[:take], w[:take]
                # Map from barycentric coords to the actual triangle in R^2
                pts = (u[:, None] * a) + (v[:, None] * b) + (w[:, None] * c)
                buf.append(pts)
                need -= take

        pts = np.vstack(buf).astype(np.float32)
        P_all.append(pts)
        # Inherit the triangle's material label for each generated point
        M_all.append(np.full((pts.shape[0],), mat_per_tri[eid], dtype=np.int32))
        E_all.append(np.full((pts.shape[0],), eid, dtype=np.int64))

    if P_all:
        X = np.vstack(P_all)
        mat_ids = np.concatenate(M_all)
        elem_ids = np.concatenate(E_all)
    else:
        # Degenerate case: no triangles or zero area
        X = np.zeros((0, 2), np.float32)
        mat_ids = np.zeros((0,), np.int32)
        elem_ids = np.zeros((0,), np.int64)

    return X, mat_ids, elem_ids


def classify_boundary_edges(points2d: np.ndarray,
                            tris: np.ndarray,
                            bbox_tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify **outer boundary edges** and classify them into
    Dirichlet (top/left/right) or Neumann (bottom).

    Algorithm:
      1) Build edge->adjacentTriangles map. Outer boundary edges have only one adjacent tri.
      2) Compute bounding box of all nodes: [xmin, xmax, ymin, ymax].
      3) For each outer edge, compute its midpoint (x,y) and compare with the bbox
         within a relative tolerance `bbox_tol * max(width, height)`:
           - y ~ ymin => Neumann (bc=1), outward normal (0,-1)
           - y ~ ymax => Dirichlet (bc=0), outward normal (0, +1)
           - x ~ xmin => Dirichlet (bc=0), outward normal (-1, 0)
           - x ~ xmax => Dirichlet (bc=0), outward normal (+1, 0)
         If an edge does not align with bbox (e.g., a hole), we default to Dirichlet
         and take a perpendicular unit vector as a (consistent but arbitrary) normal.

    Parameters
    ----------
    points2d : (N,2) float
        Node coordinates.
    tris : (M,3) int
        Triangle connectivity.
    bbox_tol : float
        Relative tolerance for bbox snapping; 1e-4 is usually safe.

    Returns
    -------
    X_bc : (Nb,2) float32
        Boundary sample points (edge midpoints).
    bc_type : (Nb,) int32
        Boundary type per point: 0=Dirichlet (L/R/T), 1=Neumann (bottom).
    n_bc : (Nb,2) float32
        Outward unit normal for each boundary point.
    bbox : (4,) float32
        Bounding box [xmin, xmax, ymin, ymax].
    """
    # Step 1: find edges with only one adjacent triangle (outer boundary)
    e2e = build_edge_adjacency(tris)
    boundary_edges = [e for e, adj in e2e.items() if len(adj) == 1]

    # Step 2: compute domain bbox
    xmin, ymin = points2d.min(axis=0)
    xmax, ymax = points2d.max(axis=0)
    extent = max(xmax - xmin, ymax - ymin)
    tol = bbox_tol * extent

    # Step 3: classify each boundary edge by comparing its midpoint with bbox planes
    mids: List[np.ndarray] = []
    bct: List[np.int32] = []
    normals: List[np.ndarray] = []
    for i, j in boundary_edges:
        p0, p1 = points2d[i], points2d[j]
        mid = 0.5 * (p0 + p1)
        x, y = mid

        if abs(y - ymin) <= tol:
            # Bottom boundary => Neumann with outward normal pointing "down"
            n = np.array([0.0, -1.0], np.float32)
            bc = np.int32(1)
        elif abs(y - ymax) <= tol:
            n = np.array([0.0,  1.0], np.float32)
            bc = np.int32(0)
        elif abs(x - xmin) <= tol:
            n = np.array([-1.0, 0.0], np.float32)
            bc = np.int32(0)
        elif abs(x - xmax) <= tol:
            n = np.array([ 1.0, 0.0], np.float32)
            bc = np.int32(0)
        else:
            # Non-bbox outer edges (e.g., holes) default to Dirichlet and use
            # a perpendicular unit vector as the "outward" normal.
            t = p1 - p0
            tlen = np.linalg.norm(t) + 1e-12
            n = np.array([-t[1]/tlen, t[0]/tlen], np.float32)
            bc = np.int32(0)

        mids.append(mid.astype(np.float32))
        bct.append(bc)
        normals.append(n)

    bbox = np.array([xmin, xmax, ymin, ymax], dtype=np.float32)
    if mids:
        return np.vstack(mids), np.asarray(bct, dtype=np.int32), np.vstack(normals), bbox
    else:
        # No boundary (degenerate); return empty arrays
        return (np.zeros((0, 2), np.float32),
                np.zeros((0,), np.int32),
                np.zeros((0, 2), np.float32),
                bbox)


# -------------------- Main Script --------------------

def main() -> None:
    """End-to-end pipeline: read VTK -> sample interior/boundary -> save NPZ -> save PNG."""
    # 1) Read VTK
    points3d, cells, cell_types, cell_scalars = parse_vtk_legacy_ascii(VTK_PATH)
    points2d = points3d[:, :2].astype(np.float64)  # we only need (x,y) for 2D Poisson/Magnetics

    # 2) Keep only triangles (VTK cell type 5)
    triangles = [c for c, t in zip(cells, cell_types) if t == 5]
    tris = np.array(triangles, dtype=np.int64)
    if tris.size == 0:
        raise RuntimeError("No triangle cells (VTK type=5) found in VTK.")

    # 3) Material per triangle (prefer 'Material' scalar; else first scalar; else zeros)
    def pick_scalar(cdict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if "Material" in cdict:
            return cdict["Material"]
        return cdict[list(cdict.keys())[0]] if len(cdict) > 0 else None

    mat_all = pick_scalar(cell_scalars)
    per_cell_mat = (np.zeros((len(cells),), dtype=np.int32)
                    if mat_all is None
                    else np.asarray(np.round(mat_all).astype(np.int32)))
    tri_mat = np.array([per_cell_mat[i] for i, t in enumerate(cell_types) if t == 5], dtype=np.int32)

    # 4) Interior sampling that avoids edges
    X_int, mat_ids, elem_ids = sample_in_triangles(points2d, tris, tri_mat,
                                                   POINTS_TOTAL, tau=TAU, seed=RANDOM_SEED,
                                                   material_weights={0:1.0, 1:1.0, 3:2.5})

    # 5) Boundary classification (bottom=Neumann; left/right/top=Dirichlet)
    X_bc, bc_type, n_bc, bbox = classify_boundary_edges(points2d, tris)

    # 6) Save NPZ
    np.savez_compressed(
        OUT_NPZ,
        X=X_int, mat_id=mat_ids, elem_id=elem_ids,
        X_bc=X_bc, bc_type=bc_type, n_bc=n_bc,
        bbox=bbox, tau=np.float32(TAU),
        unique_materials=np.unique(tri_mat)
    )

    # 7) Visualization
    fig = plt.figure(figsize=(7, 7), dpi=150)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # Draw mesh edges
    # メッシュのエッジ（均一の細線・色指定なし）
    for i, j, k in tris:
        poly = np.vstack([points2d[i], points2d[j], points2d[k], points2d[i]])
        ax.plot(poly[:, 0], poly[:, 1], c="black", linewidth=0.4)

    # Plot interior points colored by material
    uniq_mats = np.unique(mat_ids)
    mat_labels = {0: "Iron", 1: "Air", 3: "Coil"}
    for m in uniq_mats:
        mask = (mat_ids == m)
        if np.any(mask):
            ax.scatter(X_int[mask, 0], X_int[mask, 1], s=2, alpha=0.6, label=mat_labels.get(int(m)))

    # 境界点: Dirichlet=四角, Neumann=三角（矢印は描かない）
    if X_bc.shape[0] > 0:
        dir_mask = (bc_type == 0)
        neu_mask = (bc_type == 1)

        if np.any(dir_mask):
            ax.scatter(X_bc[dir_mask, 0], X_bc[dir_mask, 1],
                       s=10, marker='s', alpha=0.9, label='Dirichlet')

        if np.any(neu_mask):
            ax.scatter(X_bc[neu_mask, 0], X_bc[neu_mask, 1],s=14, marker='^', alpha=0.9, label='Neumann')
           

    ax.set_title("VTK Sampling Preview (interior + boundary)")
    ax.legend(markerscale=3, fontsize=8, loc='upper right', framealpha=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"Saved PNG -> {OUT_PNG}")
    print(f"Saved NPZ -> {OUT_NPZ}")


# Entry point
if __name__ == "__main__":
    main()
