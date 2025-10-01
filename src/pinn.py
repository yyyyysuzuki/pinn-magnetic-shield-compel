#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ -> PINN 学習（2D磁界 A_z のPoisson型）
  PDE: div( (1/mu) * grad(A) ) = - Jz
  BC : 左/右/上 = Dirichlet, 下 = Neumann
備考:
  - 材料界面ロスは入れていません（まずは動作優先）
  - コイルの Jz, 各材料の mu_r は下の辞書で設定してください
  - Neumann は下端のみとし、外向き法線 n=[0,-1] を仮定
"""
import matplotlib.pyplot as plt
import write_vtk as wv
import toolbox as tb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import time
from pathlib import Path

# ==== 入力NPZ ==========================================================
NPZ_PATH = Path("../data/pinn_points_from_vtk.npz")  # 例: あなたの生成ファイルに合わせて
# =====================================================================

# ==== 物性・ソース（必ず自案件に合わせて調整） ==========================
mu0 = 4e-7 * np.pi  # 真空透磁率 [H/m]
at  = 2000
coils = 40*1e-3*40*1e-3
jo  = at / coils
# ==== 正規化設定 =====================================================
astar = 1e+6
jstar = 1.0
nustar = jstar/astar

# 材料ID -> 相対透磁率 mu_r
MU_R = {
    0: 1.0/nustar,   # 例: iron  (実験的に鉄を空気に変更しました)
    1: 1.0/nustar,      # 例: air
    3: 1.5/nustar,      # 例: coil  (巻線領域は近似的に空気扱いにしておく)
}
# 材料ID -> Jz [A/m^2]（コイル領域のみ非ゼロにする例）
JZ = {
    0: 0.0,      # iron
    1: 0.0,      # air
    3: jo,      # coil (←案件の電流密度に合わせて！)
}
# Dirichlet 固定値（境界上Aの値）: ここでは 0 に固定
DIRICHLET_VALUE = 0.0
# Neumann 指定値 q = n · ((1/mu) grad A)（ここでは 0 に固定）
NEUMANN_VALUE = 0.0
# =====================================================================

# ==== 学習ハイパラ =====================================================
HIDDEN = 64
DEPTH  = 4
LR     = 1e-3
EPOCHS = 1000000
LAMBDA_PDE = 1.0
LAMBDA_DIR = 1e+5
LAMBDA_NEU = 1e-5
BATCH_INT = 8192       # 内部点のバッチ（メモリに応じて）
BATCH_BC  = 4096       # 境界点のバッチ
SEED = 42
# =====================================================================

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== データ読み込み ====================================================
npz = np.load(NPZ_PATH, allow_pickle=True)
X_np      = npz["X"].astype(np.float32)            # (N,2)
mat_np    = npz["mat_id"].astype(np.int64)         # (N,) 所属する要素の材料番号
elem_np   = npz["elem_id"].astype(np.int64)        # (N,) 所属する要素番号
Xbc_np    = npz["X_bc"].astype(np.float32) if "X_bc" in npz.files else np.zeros((0,2), np.float32) #(N_bc,2) 境界条件座標点
bc_type_np= npz["bc_type"].astype(np.int64) if "bc_type" in npz.files else np.zeros((0,), np.int64) # (N_bc,) 0 mean Dir. 1 mean Neu.

# tensors
X      = torch.from_numpy(X_np).to(device)                   # (N,2)
mat_id = torch.from_numpy(mat_np).to(device)                 # (N,)
X_bc   = torch.from_numpy(Xbc_np).to(device)                 # (Nb,2)
bc_type= torch.from_numpy(bc_type_np).to(device)             # (Nb,)

# 材料ごとの 1/mu と Jz を点ごとに持たせる
mu_r_arr = np.vectorize(lambda m: MU_R.get(int(m), 1.0))(mat_np).astype(np.float32) #vectorizeでmat_npを順にMU_R.getに代入
inv_mu_np = (1.0 / (mu0 * mu_r_arr)).astype(np.float32)      # (N,)
Jz_np     = np.vectorize(lambda m: JZ.get(int(m), 0.0))(mat_np).astype(np.float32)
inv_mu = torch.from_numpy(inv_mu_np).to(device)              # (N,)
Jz     = torch.from_numpy(Jz_np).to(device)                  # (N,)

# ==== NN モデル =========================================================
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, width=64, depth=4):
        super().__init__()
        layers = []
        dims = [in_dim] + [width]*depth + [out_dim]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        # He/Xavier 初期化（簡易）
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

model = MLP(2,1,HIDDEN,DEPTH).to(device)
opt   = optim.Adam(model.parameters(), lr=LR)

# ==== 微分ユーティリティ ===============================================
def grad(outputs, inputs):
    """∂outputs/∂inputs を autograd で計算（outputs: (N,1), inputs: (N,2))"""
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

def pde_residual(batch_X, batch_inv_mu, batch_Jz):
    """
    r = div( (1/mu) grad A ) + Jz
    """
    batch_X.requires_grad_(True)
    A = model(batch_X)                     # (B,1)
    g = grad(A, batch_X)                  # (B,2) -> grad A = [dA/dx, dA/dy]
    q = batch_inv_mu.unsqueeze(1) * g     # (B,2) -> (1/mu) * grad A
    # div q:
    dqdx = grad(q[:, :1], batch_X)[:, :1] # ∂qx/∂x
    dqdy = grad(q[:, 1:], batch_X)[:, 1:] # ∂qy/∂y
    divq = dqdx + dqdy                    # (B,1)
    r = divq + batch_Jz.unsqueeze(1)      # (B,1)   （符号は課題に合わせて調整）
    return r

# ==== BC ロス ===========================================================
def dirichlet_loss(Xd):
    if Xd.numel() == 0:
        return torch.tensor(0.0, device=device)
    A = model(Xd)
    return torch.mean((A - DIRICHLET_VALUE)**2)

def neumann_loss(Xn):
    if Xn.numel() == 0:
        return torch.tensor(0.0, device=device)
    # 下端のみ → 外向き法線 n = [0, -1]
    Xn.requires_grad_(True)
    A = model(Xn)
    g = grad(A, Xn)                 # (B,2)
    # 材料IDが無いので、Neumannでも (1/mu) は空気想定か一定で近似…は微妙。
    # 厳密にやるなら Xn を最近傍の内部点材料へマップする等が必要。
    # ここでは簡単化として (1/mu)=1/mu0 とします（必要に応じて改良）。
    inv_mu_bc = 1.0 / mu0
    flux_n = inv_mu_bc * (-g[:, 1:])  # n=[0,-1] なので n·gradA = -dA/dy
    return torch.mean((flux_n - NEUMANN_VALUE)**2)

# ==== データ分割（Dirichlet/Neumann） ================================
if X_bc.shape[0] > 0:
    dir_mask = (bc_type == 0)
    neu_mask = (bc_type == 1)
    Xd_all = X_bc[dir_mask]
    Xn_all = X_bc[neu_mask]
else:
    Xd_all = torch.zeros((0,2), device=device)
    Xn_all = torch.zeros((0,2), device=device)

# ==== 学習ループ =======================================================
N = X.shape[0]
Nb_d = Xd_all.shape[0]
Nb_n = Xn_all.shape[0]

print(f"[info] interior N={N}, Dirichlet Nb={Nb_d}, Neumann Nb={Nb_n}, device={device}")
epochhistory = []
losspde  = []
lossdir  = []
lossneu  = []

for ep in range(1, EPOCHS+1):
    start = time.time()
    model.train()
    # ---- interior: ミニバッチ ----
    idx_int = torch.randint(0, N, (min(BATCH_INT, N),), device=device)
    Xi  = X[idx_int]
    inv = inv_mu[idx_int]
    J   = Jz[idx_int]
    r   = pde_residual(Xi, inv, J)
    loss_pde = torch.mean(r**2)

    # ---- boundary: ミニバッチ ----
    if Nb_d > 0:
        idxd = torch.randint(0, Nb_d, (min(BATCH_BC, Nb_d),), device=device)
        loss_dir = dirichlet_loss(Xd_all[idxd])
    else:
        loss_dir = torch.tensor(0.0, device=device)

    if Nb_n > 0:
        idxn = torch.randint(0, Nb_n, (min(BATCH_BC, Nb_n),), device=device)
        loss_neu = neumann_loss(Xn_all[idxn])
    else:
        loss_neu = torch.tensor(0.0, device=device)

    loss = LAMBDA_PDE*loss_pde + LAMBDA_DIR*loss_dir + LAMBDA_NEU*loss_neu
    epochhistory.append(ep)
    losspde.append((LAMBDA_PDE*loss_pde).item())
    lossdir.append((LAMBDA_DIR*loss_dir).item())
    lossneu.append((LAMBDA_NEU*loss_neu).item())

    opt.zero_grad()
    loss.backward()
    opt.step()

    if ep % 5000 == 0 or ep == 1:
        print(f"[{ep:5d}] loss={loss.item():.3e} | pde={loss_pde.item():.3e} dir={loss_dir.item():.3e} neu={loss_neu.item():.3e}")

        model.eval()

        # エクスポート実行
        wv.export_vtk_from_json(
            json_path="../data/gen_0ind_0_mesh_A_Bx_By.json",   # あなたのJSON
            vtk_out=f"../results/{ep}_A_B_vectors.vtk",
            model=model,
            device=device,
            batch=16384,
            index_base=0,  # JSONのノード番号が0起点なら0、1起点なら1
        )

        if ep % 10000 == 0 or ep == 1:
            tb.write_four_lists_to_csv(epochhistory,losspde, lossdir, lossneu, f"../results/loss.csv", ("epoch","pde","dir","neu"))

        # ごりおし可視化
        # bbox 推定（npz に bbox が入っていればそれを使う）
        npz = np.load("../data/pinn_points_from_vtk.npz", allow_pickle=True)
        xmin,xmax,ymin,ymax = npz["bbox"]
        nx, ny = 400, 400  # 解像度
        xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(xs, ys)
        XY = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

        # バッチでモデル推論
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xy_t = torch.from_numpy(XY).to(device)
        A_flat=[]
        with torch.no_grad():
            for s in range(0, xy_t.shape[0], 16384):
                A_flat.append(model(xy_t[s:s+16384]).squeeze(1).cpu().numpy())
        A_flat = np.concatenate(A_flat)
        AA = A_flat.reshape(ny, nx)

        # 等値線描画（領域外にも色が出る点に注意）
        fig = plt.figure(figsize=(7,6), dpi=150)
        ax = plt.gca(); im = ax.contourf(XX, YY, AA, levels=30)
        fig.colorbar(im, ax=ax, label="A_z")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("A distribution (grid contourf)")
        fig.tight_layout(); fig.savefig(f"../results/{ep}_A_grid_contourf.png")
        print(f"{ep}_saved A_grid_contourf.png")


torch.save(model.state_dict(), f"../results/pinn_model.pt")
print(f"[OK] saved -> {ep}_pinn_model.pt")

end = time.time()

print("実行時間：",end-start,"秒")

