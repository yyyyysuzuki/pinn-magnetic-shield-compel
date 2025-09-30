from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
import json
import csv

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):  # アイテムがリストなら、再帰的に展開
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def read_Json(fp):
    with open(fp, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_csv(file_path):
    data = []
    with open(file_path, newline='') as file:
        reader = csv.reader(file)  # タブ区切りの場合、delimiter='\t'を使用
        for row in reader:
            data.append([int(value) for value in row])
    return data

def plot_learning_curve(file_path, loss_values):
    # 学習曲線をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(range(len(loss_values)))  # x軸の目盛りを設定
    plt.grid()

    # グラフをファイルに保存
    plt.savefig(file_path)  # 指定されたファイル名で保存
    plt.close()  # プロットを閉じる

def plot_predictions(true_labels, predicted_values, file_path):
    # R²を計算
    r_squared = r2_score(true_labels, predicted_values)

    # プロット
    plt.figure(figsize=(8, 8))
    plt.scatter(true_labels, predicted_values, color='blue', label='Predicted values')
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 
             color='red', linestyle='--', label='y=x line')

    # ラベルとタイトルの設定
    plt.xlabel('Prediction')
    plt.ylabel('Labels')
    plt.title(f'Prediction vs True Labels (R² = {r_squared:.2f})')
    plt.legend()
    plt.grid()
    plt.axis('equal')  # x軸とy軸のスケールを同じにする
    plt.xlim(min(true_labels) - 1, max(true_labels) + 1)
    plt.ylim(min(true_labels) - 1, max(true_labels) + 1)

    # プロットをファイルに保存
    plt.savefig(file_path)
    plt.close()  # プロットを閉じる

def write_two_lists_to_csv(list1, list2, filename, header):
    if len(list1) != len(list2):
        raise ValueError("リストの長さが一致していません")
    
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for v1, v2 in zip(list1, list2):
            writer.writerow([v1, v2])

def write_three_lists_to_csv(list1, list2, list3, filename, header):
    if len(list1) != len(list2) or len(list1) != len(list3):
        raise ValueError("リストの長さが一致していません")
    
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for v1, v2, v3 in zip(list1, list2, list3):
            writer.writerow([v1, v2, v3])

def write_four_lists_to_csv(list1, list2, list3, list4, filename, header):
    if not (len(list1) == len(list2) == len(list3) == len(list4)):
        raise ValueError("リストの長さが一致していません")
    
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for v1, v2, v3, v4 in zip(list1, list2, list3, list4):
            writer.writerow([v1, v2, v3, v4])


def write_five_lists_to_csv(list1, list2, list3, list4, list5, filename, header):
    if not (len(list1) == len(list2) == len(list3) == len(list4) == len(list5)):
        raise ValueError("リストの長さが一致していません")
    
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for v1, v2, v3, v4, v5 in zip(list1, list2, list3, list4, list5):
            writer.writerow([v1, v2, v3, v4, v5])

def relative_error_loss_pyg(pred, target, batch, epsilon, max_error=10):
    """
    pred:   [num_nodes, 1] or [num_nodes]
    target: [num_nodes, 1] or [num_nodes]
    batch:  [num_nodes] （各ノードが属するバッチ番号）
    """
    pred = pred.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)

    # 相対誤差を計算（真値が小さすぎる場合の不安定さを回避）
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

    # 大きすぎる相対誤差を抑制（例: 外れ値の影響を制限）
    rel_error = torch.clamp(rel_error, min=0.0, max=max_error)

    num_graphs = batch.max().item() + 1
    graph_errors = []

    for i in range(num_graphs):
        mask = (batch == i)
        if mask.sum() > 0:
            mean_error = rel_error[mask].mean()
            graph_errors.append(mean_error)
    
    return torch.stack(graph_errors).mean(),graph_errors

def relative_loss(y_pred, y_true, epsilon=1e-15, max_rel_error=10.0):
    """
    y_pred: 予測値 (Tensor)
    y_true: 真値 (Tensor)
    epsilon: 真値が0に近い時の除算安定化項
    max_rel_error: 相対誤差の上限値（clamp）
    """

    # 相対誤差を計算（真値が小さすぎる場合の不安定さを回避）
    rel_error = torch.abs(y_pred - y_true) / (torch.abs(y_true) + epsilon)

    # 大きすぎる相対誤差を抑制（例: 外れ値の影響を制限）
    rel_error = torch.clamp(rel_error, min=0.0, max=max_rel_error)

    # 平均誤差として出力
    return rel_error.mean()

def mse_loss_pyg(pred, target, batch):
    """
    pred:   [num_nodes, 1] or [num_nodes]
    target: [num_nodes, 1] or [num_nodes]
    batch:  [num_nodes] （各ノードが属するバッチ番号）
    """
    pred = pred.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)

    squared_error = (pred - target) ** 2

    num_graphs = batch.max().item() + 1
    graph_errors = []

    for i in range(num_graphs):
        mask = (batch == i)
        if mask.sum() > 0:
            mean_error = squared_error[mask].mean()
            graph_errors.append(mean_error)
    
    return torch.stack(graph_errors).mean(), graph_errors
