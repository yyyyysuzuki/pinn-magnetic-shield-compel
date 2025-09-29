#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ -> サンプリング個数のCSV集計
--------------------------------
前処理で作成した PINN 用 NPZ から、属性ごとのサンプル個数を CSV に出力します。

入力 (NPZ に想定される配列):
- X:        (N,2) 内部サンプル点（使わないが存在前提ではない）
- mat_id:   (N,)   内部サンプル点の材料ID
- X_bc:     (Nb,2) 境界サンプル点（無いこともある）
- bc_type:  (Nb,)  境界種別ID（0=Dirichlet, 1=Neumann）
  ※ これらが無くてもスクリプトは落ちず、ある分だけ集計します。

出力:
- CSV (デフォルト: sampling_counts.csv)
    列: attribute_group, attribute_id, attribute_name, count

使い方例:
  python sampling_counts_from_npz.py \
      --npz pinn_points_from_vtk.npz \
      --out sampling_counts.csv \
      --mat-map '{"0":"air","1":"iron","2":"coil"}'

依存:
  pip install numpy pandas
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_npz(npz_path: Path) -> dict:
    """NPZ を読み込んで dict にして返す（allow_pickle=True）。"""
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def parse_mat_map(arg: str | None) -> dict[int, str]:
    """--mat-map のJSON文字列を dict[int,str] に変換。未指定なら空dict。"""
    if not arg:
        return {}
    try:
        raw = json.loads(arg)
        # キーが文字列でも int に直す
        return {int(k): str(v) for k, v in raw.items()}
    except Exception as e:
        raise SystemExit(f"--mat-map のJSON解釈に失敗: {e}")


def summarize_counts(data: dict, mat_name_map: dict[int, str]) -> pd.DataFrame:
    """
    材料別（内部点）と境界種別（外周点）のカウントをDataFrameで返す。
    - attribute_group: "material" or "boundary"
    - attribute_id:    int  (材料ID or 0/1)
    - attribute_name:  文字列（マップがあればその名前）
    - count:           個数
    """
    rows: list[dict] = []

    # --- 材料（内部点） ---
    if "mat_id" in data:
        mat_ids = np.asarray(data["mat_id"]).ravel()
        if mat_ids.size > 0:
            uniq, cnt = np.unique(mat_ids, return_counts=True)
            for mid, c in zip(uniq, cnt):
                mid_int = int(mid)
                rows.append({
                    "attribute_group": "material",
                    "attribute_id": mid_int,
                    "attribute_name": mat_name_map.get(mid_int, f"material_{mid_int}"),
                    "count": int(c),
                })

    # --- 境界（Dirichlet/Neumann）---
    if "bc_type" in data and "X_bc" in data:
        bc_type = np.asarray(data["bc_type"]).ravel()
        if bc_type.size > 0:
            uniq, cnt = np.unique(bc_type, return_counts=True)
            for bt, c in zip(uniq, cnt):
                bt_int = int(bt)
                name = "Dirichlet" if bt_int == 0 else "Neumann" if bt_int == 1 else f"type_{bt_int}"
                rows.append({
                    "attribute_group": "boundary",
                    "attribute_id": bt_int,
                    "attribute_name": name,
                    "count": int(c),
                })
        else:
            rows.append({
                "attribute_group": "boundary",
                "attribute_id": None,
                "attribute_name": "None",
                "count": 0,
            })
    else:
        # 境界情報がNPZに入っていない場合
        rows.append({
            "attribute_group": "boundary",
            "attribute_id": None,
            "attribute_name": "None",
            "count": 0,
        })

    df = pd.DataFrame(rows, columns=["attribute_group", "attribute_id", "attribute_name", "count"])
    # 表示のためソート（任意）
    df = df.sort_values(by=["attribute_group", "attribute_id"], kind="stable").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="NPZから属性ごとのサンプル数をCSV出力")
    ap.add_argument("--npz", type=Path, required=True, help="入力NPZパス（pinn_points_*.npz）")
    ap.add_argument("--out", type=Path, default=Path("sampling_counts.csv"), help="出力CSVパス")
    ap.add_argument(
        "--mat-map",
        type=str,
        default=None,
        help='材料ID→名称のJSON文字列（例: \'{"0":"air","1":"iron","2":"coil"}\'）'
    )
    args = ap.parse_args()

    data = load_npz(args.npz)
    mat_name_map = parse_mat_map(args.mat_map)
    df = summarize_counts(data, mat_name_map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] wrote CSV -> {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
