#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONメッシュと学習済みPINNモデルから、Paraviewで可視化可能なVTK(legacy ASCII, UNSTRUCTURED_GRID)を出力するユーティリティ。

流れ:
  1) JSONから Node / Element を読み込み
  2) ノード座標 (x,y) をモデルに入力して A(x,y) をバッチ推論（Aは節点スカラー）
  3) 三角形要素ごとに B=(Bx,By) を計算（要素ベクトル）
  4) VTK ファイルとして A(POINT_DATA), B(CELL_DATA) を書き出し

前提:
  - Node は少なくとも [x, y] の2列を持つ（3列目以降は任意: 例 Aの初期値など）
  - Element は [material_id, n1, n2, n3] の4列（n* はノード番号）
  - ノード番号の起点（0/1）は引数 index_base で調整可能（デフォルト=0起点）
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import json, os
import numpy as np
import torch
import csv


# -------------------------
# 入出力まわり
# -------------------------
def read_json_mesh(fp: str | os.PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    JSONファイルから Node / Element を読み込む。

    期待するJSON構造:
      {
        "Node":    [[x,y,(...任意の列)], ...],
        "Element": [[material_id, n1, n2, n3], ...]
      }

    戻り値:
      NodeData    : (N, >=2) float64  先頭2列が (x,y)
      ElementData : (M, 4)   int64    [material, n1, n2, n3]
    """
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    NodeData = np.array(data["Node"], dtype=float)
    ElementData = np.array(data["Element"], dtype=int)
    return NodeData, ElementData


def write_vtk_unstructured(
    filename: str | os.PathLike,
    A_node: np.ndarray,
    Bx_cell: np.ndarray,
    By_cell: np.ndarray,
    NodeData: np.ndarray,
    ElementData: np.ndarray,
) -> None:
    """
    VTK (legacy ASCII, UNSTRUCTURED_GRID) を出力する。

    出力フィールド:
      - POINT_DATA: SCALARS A        (節点スカラー)
      - CELL_DATA : VECTORS BxBy     (要素ベクトル)

    引数:
      filename   : 出力先パス
      A_node     : (N,) or (N,1)  節点ごとの A 値（float）
      Bx_cell    : (M,)           要素ごとの Bx
      By_cell    : (M,)           要素ごとの By
      NodeData   : (N,>=2)        先頭2列が (x,y)。3列目があれば z として出す
      ElementData: (M,4)          [material, n1, n2, n3]
    """
    A_node = np.asarray(A_node).reshape(-1)
    Bx_cell = np.asarray(Bx_cell).reshape(-1)
    By_cell = np.asarray(By_cell).reshape(-1)

    N = NodeData.shape[0]
    M = ElementData.shape[0]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Title Data\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # --- POINTS ---
        f.write(f"POINTS {N} float\n")
        has_z = NodeData.shape[1] >= 3
        for i in range(N):
            x, y = float(NodeData[i, 0]), float(NodeData[i, 1])
            z = float(NodeData[i, 2]) if has_z else 0.0
            f.write(f"{x:.16f} {y:.16f} {z:.16f}\n")

        # --- CELLS（三角形のみ想定）---
        f.write(f"CELLS {M} {4 * M}\n")
        for e in range(M):
            n1, n2, n3 = int(ElementData[e, 1]), int(ElementData[e, 2]), int(ElementData[e, 3])
            f.write(f"3 {n1} {n2} {n3}\n")

        # --- CELL_TYPES (5=triangle) ---
        f.write(f"CELL_TYPES {M}\n")
        for _ in range(M):
            f.write("5\n")

        # --- POINT_DATA: A ---
        f.write(f"POINT_DATA {N}\n")
        f.write("SCALARS A float\n")
        f.write("LOOKUP_TABLE default\n")
        for val in A_node:
            f.write(f"{float(val):.16f}\n")

        # --- CELL_DATA: BxBy ---
        f.write(f"CELL_DATA {M}\n")
        f.write("VECTORS BxBy float\n")
        for e in range(M):
            f.write(f"{float(Bx_cell[e]):.16f} {float(By_cell[e]):.16f} 0.0\n")


# -------------------------
# 物理量の計算
# -------------------------
def eval_A_on_nodes(
    model: torch.nn.Module,
    NodeData: np.ndarray,
    batch: int = 16384,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    ノード座標 (x,y) をモデルに入力して A(x,y) を推論する。

    引数:
      model   : 学習済みPINNモデル（forward: (B,2) -> (B,1) を想定）
      NodeData: (N,>=2) 先頭2列が (x,y)
      batch   : 推論時のバッチサイズ
      device  : CUDA / CPU。None のときは自動判定

    戻り値:
      A: (N,) float32  各節点での A 値
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    xy = NodeData[:, :2].astype(np.float32)
    xy_t = torch.from_numpy(xy).to(device)

    outs = []
    with torch.no_grad():
        for s in range(0, xy_t.shape[0], batch):
            out = model(xy_t[s:s+batch]).squeeze(1)  # (b,1) -> (b,)
            outs.append(out.detach().cpu().numpy())
    A = np.concatenate(outs).astype(np.float32)
    return A


def compute_B_on_triangles(
    A_node: np.ndarray,
    NodeData: np.ndarray,
    ElementData: np.ndarray,
    index_base: int = 0,
    eps_area: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    三角形要素ごとに B=(Bx,By) を計算する。
    いわゆる一次(線形)三角形の補間で、A の節点値から要素内の一定 B を算出する定式化。

    引数:
      A_node     : (N,)   節点スカラー A
      NodeData   : (N,>=2) 先頭2列が (x,y)
      ElementData: (M,4)  [material_id, n1, n2, n3]
      index_base : 要素のノード番号の起点（0 or 1）※JSONの定義に合わせて設定
      eps_area   : 面積が極小のときのゼロ割り防止用イプシロン

    戻り値:
      (Bx, By): ともに (M,) 要素ごとの磁束密度ベクトルの成分
    """
    A = np.asarray(A_node).reshape(-1)
    N = NodeData.shape[0]
    M = ElementData.shape[0]

    if index_base not in (0, 1):
        raise ValueError("index_base は 0 か 1 を指定してください。")

    # 出力配列
    Bx = np.zeros(M, dtype=float)
    By = np.zeros(M, dtype=float)

    for ele in range(M):
        # materials は今は使わないが将来の拡張のために読み出しておく
        # （例: 要素内の mu をここで使って B→H 変換など）
        _material = int(ElementData[ele, 0])

        # 要素の3節点番号を取得し、index_base に応じて0起点へ正規化
        e = ElementData[ele, 1:4].astype(int)
        if index_base == 1:
            e = e - 1  # 1起点→0起点に補正

        # 安全チェック
        if np.any(e < 0) or np.any(e >= N):
            raise IndexError(f"要素{ele}の節点番号が範囲外です: {e}")

        # 係数ベクトル c, d を構築（各iに対し、j,k は残りの2頂点）
        c = np.zeros(3, dtype=float)
        d = np.zeros(3, dtype=float)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            # y座標差 / x座標差
            c[i] = NodeData[e[j], 1] - NodeData[e[k], 1]   # Δy
            d[i] = NodeData[e[k], 0] - NodeData[e[j], 0]   # Δx

        # 三角形面積 (2*Area) に相当する量を算出してから 1/2
        det = (c[0]*d[1] - c[1]*d[0]) * 0.5
        area = abs(det)
        if area < eps_area:
            # 面積が極小のときはゼロ割りを避けて0ベクトルに（メッシュ品質要確認）
            continue

        # 要素平均の B を計算（線形要素における一定Bの形）
        # Bx = (1/(2*Area)) * Σ d_i * A_i,  By = -(1/(2*Area)) * Σ c_i * A_i
        inv2A = 1.0 / (2.0 * area)
        # ただし上で det=0.5*(...) としているため inv2A = 1/(2A) = 1/(2*abs(det))
        inv2A = 1.0 / (2.0 * area)

        # 係数は det の符号で向きが反転するが、Areaを絶対値にしたため、
        # 幾何の向きに依らず大きさは安定。向きを厳密に扱いたければ det の符号を使う実装に変更可。
        for i in range(3):
            Bx[ele] += d[i] * float(A[e[i]]) * inv2A
            By[ele] -= c[i] * float(A[e[i]]) * inv2A

    return Bx, By


# -------------------------
# 1発エクスポート関数
# -------------------------
def export_vtk_from_json(
    json_path: str | os.PathLike,
    vtk_out: str | os.PathLike,
    model: torch.nn.Module,
    *,
    device: Optional[torch.device] = None,
    batch: int = 16384,
    index_base: int = 0,
) -> None:
    """
    JSONメッシュと学習済みモデルから VTK を一気に生成する高水準API。

    引数:
      json_path : 入力JSONのパス
      vtk_out   : 出力VTKのパス
      model     : 学習済みPINNモデル（forward: (B,2)->(B,1)）
      device    : 推論デバイス（Noneなら自動判定）
      batch     : 推論バッチサイズ
      index_base: 要素のノード番号起点（0 or 1）
    """
    # 1) JSONロード
    NodeData, ElementData = read_json_mesh(json_path)

    # 2) ノード座標で A を推論
    A_node = eval_A_on_nodes(model, NodeData, batch=batch, device=device)  # (N,)

    # 3) 要素ごとに B を計算
    Bx, By = compute_B_on_triangles(A_node, NodeData, ElementData, index_base=index_base)

    # 4) VTK出力
    write_vtk_unstructured(vtk_out, A_node, Bx, By, NodeData, ElementData)


# -------------------------
# 使い方の例（コメント）
# -------------------------
"""
# 学習済みモデル定義（あなたの学習スクリプトと同じネットワークを再構築）
class MLP(torch.nn.Module):
    def __init__(self, in_dim=2, out_dim=1, width=64, depth=4):
        super().__init__()
        layers=[]; dims=[in_dim]+[width]*depth+[out_dim]
        for i in range(len(dims)-2):
            layers += [torch.nn.Linear(dims[i], dims[i+1]), torch.nn.Tanh()]
        layers += [torch.nn.Linear(dims[-2], dims[-1])]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# モデル読込
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
model.load_state_dict(torch.load("pinn_model.pt", map_location=device))
model.eval()

# エクスポート実行
export_vtk_from_json(
    json_path="gen_0ind_0_mesh_A_Bx_By.json",   # あなたのJSON
    vtk_out="A_B_vectors.vtk",
    model=model,
    device=device,
    batch=16384,
    index_base=0,  # JSONのノード番号が0起点なら0、1起点なら1
)
"""
def write_three_lists_to_csv(list1, list2, list3, filename, header):
    if len(list1) != len(list2) or len(list1) != len(list3):
        raise ValueError("リストの長さが一致していません")
    
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for v1, v2, v3 in zip(list1, list2, list3):
            writer.writerow([v1, v2, v3])