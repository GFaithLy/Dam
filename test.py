import datetime
import torch
import warnings
from sklearn import metrics
import scipy
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Dataset, DataLoader
import pickle
import numpy as np
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.nn import AttentiveFP, Linear, MLP, aggr
import random
from torch_scatter import scatter_sum, scatter_max
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import csv
import torch.nn as nn
# 导入FastKAN相关模块
from kan.fastkan import FastKANLayer
# Swin Transformer
from Swin_Transformer.model20 import swin_smaller_patch4_window5_1

# ====================== 基础配置（用户需根据实际环境修改）====================== #
# 设备设置（自动检测GPU，无GPU则使用CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {device}")

# 模型路径配置（5个训练好的模型路径，需替换为实际保存路径）
MODEL_PATHS = [
    "/model/best_model_fold_1_img20.pt",
    "/model/best_model_fold_2_img20.pt",
    "/model/best_model_fold_3_img20.pt",
    "model/best_model_fold_4_img20.pt",
    "/model/best_model_fold_5_img20.pt"
]

# 测试数据集配置（支持单数据集或多数据集，用户需替换为实际路径）
# 格式：{"数据集名称": "数据集pkl文件路径"}
TEST_DATASETS = {
    "新测试集High": "/opt/data/private/two/data/Hightest.pkl",

}

# 输出配置（测试结果保存路径，自动创建目录）
OUTPUT_ROOT = "/Hightest_results"  # 结果根目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)
BATCH_SIZE = 16  # 测试批量大小（根据GPU内存调整，建议与训练时一致）
SEED = 100  # 随机种子（确保数据加载顺序一致）

# 固定图像尺寸（与训练一致，仅支持20×20）
IMG_SIZE = 20

# 忽略无关警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ====================== 1. 工具函数定义（与训练代码保持一致）====================== #
class PLBA_Dataset(Dataset):
    """数据集类：与训练代码完全一致，用于加载测试数据"""
    def __init__(self, *args):
        if args[0] == "file":
            filepath = args[1]
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"数据集文件不存在: {filepath}")
            with open(filepath, 'rb') as f:
                self.G_list = pickle.load(f)
            self.len = len(self.G_list)
        elif args[0] == 'list':
            self.G_list = args[1]
            self.len = len(args[1])

    def __getitem__(self, index):
        G = self.G_list[index]
        G[5] = torch.tensor(G[5], dtype=torch.float32)  # 确保标签为float32
        return G[0], G[1], G[2], G[3], G[5]

    def __len__(self):
        return self.len


def set_gpu(data, device):
    """将数据移动到指定设备（GPU/CPU）"""
    data_gpu = []
    for g in data:
        data_gpu.append(g.to(device))
    return data_gpu


def metrics_reg(targets, predicts):
    """回归任务评估指标计算：MAE、RMSE、Pearson R、SD、Spearman S、CI"""
    # 转换为numpy数组
    targets = np.array(targets)
    predicts = np.array(predicts)
    
    # 1. 平均绝对误差（MAE）
    mae = np.mean(np.abs(targets - predicts))
    
    # 2. 均方根误差（RMSE）
    rmse = np.sqrt(np.mean((targets - predicts) ** 2))
    
    # 3. Pearson相关系数
    r, _ = scipy.stats.mstats.pearsonr(targets, predicts)
    r = 0 if np.isnan(r) else r  # 处理NaN情况
    
    # 4. 标准偏差（SD）
    lr = LinearRegression()
    lr.fit(predicts.reshape(-1, 1), targets)
    y_hat = lr.predict(predicts.reshape(-1, 1))
    sd = np.sqrt(np.sum((targets - y_hat) ** 2) / (len(targets) - 1)) if len(targets) > 1 else 0.0
    
    # 5. Spearman相关系数
    s, _ = spearmanr(targets, predicts)
    s = 0 if np.isnan(s) else s
    
    # 6. 决定系数（CI）
    ci = 1 - (np.sum((targets - predicts) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
    ci = max(ci, 0)  # CI不能为负
    
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Pearson R": round(r, 4),
        "SD": round(sd, 4),
        "Spearman S": round(s, 4),
        "CI": round(ci, 4)
    }


# ====================== 2. 模型结构定义（与训练代码完全一致）====================== #
# 边类型感知的AttentiveFP模块
class EdgeAwareAttentiveFP(MessagePassing):
    def __init__(self, hidden_channels, edge_dim, num_layers=3, num_timesteps=3):
        super().__init__(aggr='add', flow='source_to_target')
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps

        self.node_encoder = Linear(-1, hidden_channels)
        self.edge_encoder = MLP([edge_dim, hidden_channels//2, hidden_channels], dropout=0.1)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Sequential(
                Linear(3 * hidden_channels, hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels)
            ))

        self.att = Linear(3 * hidden_channels, 1)
        self.timestep_lins = nn.ModuleList()
        for _ in range(num_timesteps - 1):
            self.timestep_lins.append(Linear(hidden_channels, hidden_channels))
        self.out_proj = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = x + F.relu(self.propagate(edge_index, x=x, edge_emb=edge_emb, conv=conv))

        xs = [x]
        for lin in self.timestep_lins:
            x = F.relu(lin(x) + self.propagate(edge_index, x=x, edge_emb=edge_emb, conv=self.convs[-1]))
            xs.append(x)
        x = torch.mean(torch.stack(xs, dim=0), dim=0)

        x = self.out_proj(x)
        return global_add_pool(x, batch)

    def message(self, x_i, x_j, edge_emb, conv):
        att_input = torch.cat([x_i, x_j, edge_emb], dim=-1)
        alpha = F.softmax(self.att(att_input), dim=0)
        message = torch.cat([x_j, edge_emb, x_i], dim=-1)
        message = conv(message)
        return alpha * message


# 边权重提取器
class ExtractorMLP(nn.Module):
    def __init__(self, in_dim, bias=True):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, in_dim*2, bias), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(in_dim*2, 1, bias)
        )

    def forward(self, emb):
        return self.feature_extractor(emb)


# 多模态注意力模块
class MultiModalAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.node_att = EdgeAwareAttentiveFP(
            hidden_channels=hidden_channels,
            edge_dim=edge_dim,
            num_layers=3,
            num_timesteps=3
        )
        self.final_proj = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        graph_feat = self.node_att(x, edge_index, edge_attr, batch)
        return self.final_proj(graph_feat)


# 交互图模块（仅适配20×20图像）
class InterGNN(torch.nn.Module):
    def __init__(self, metadata, edge_dim, hidden_channels, out_channels, num_layers, initial_edge_repr_dim=None, img_size=20):
        super().__init__()
        self.edge_mlp = MLP([128 + 8, 512, 64, 16], dropout=0.1)
        self.lin_mpl = Linear(16, 16)
        self.edge_lin = Linear(1, 8)
        
        # 【修复核心】先给self.img_size赋值，再进行尺寸校验
        self.img_size = img_size  # 这行移到校验前面
        # 强制校验：仅允许20×20尺寸
        if self.img_size != 20:
            raise ValueError(f"Swin模块仅支持20×20图像，当前输入尺寸：{self.img_size}×{self.img_size}")
        
        # 后续代码不变...
        self.swin_transformer = swin_smaller_patch4_window5_1(pretrained=False)
        self.swin_transformer.layers[0].blocks[0].window_size = 5
        self.swin_transformer.layers[1].blocks[0].window_size = 5
        self.swin_transformer.head = nn.Identity()
        self.swin_proj = Linear(self.swin_transformer.num_features, hidden_channels)
        # 图池化
        self.global_pool_ligand = aggr.AttentionalAggregation(gate_nn=nn.Linear(hidden_channels, 1))
        self.global_pool_protein = aggr.AttentionalAggregation(gate_nn=nn.Linear(hidden_channels, 1))
        
        # KAN层
        self.fastkan_ligand1 = FastKANLayer(input_dim=18, output_dim=hidden_channels, num_grids=14)
        self.fastkan_ligand2 = FastKANLayer(input_dim=hidden_channels, output_dim=hidden_channels, num_grids=14)
        self.fastkan_protein1 = FastKANLayer(input_dim=18, output_dim=hidden_channels, num_grids=14)
        self.fastkan_protein2 = FastKANLayer(input_dim=hidden_channels, output_dim=hidden_channels, num_grids=14)
        
        # 固定生成20×20图像特征
        self.target_dim = 20 * 20
        self.feat_to_img = nn.Sequential(
            Linear(2 * hidden_channels, 256),
            nn.ReLU(),
            Linear(256, 3 * self.target_dim),
            nn.Unflatten(1, (3, 20, 20))
        )
        
        self.edge_types = metadata[1]
        self.edge_weight_extractors = nn.ModuleDict()
        self.register_buffer('m_out_mean', torch.zeros(32))
        self.mean_initialized = False

        if initial_edge_repr_dim is not None:
            for edge_type in self.edge_types:
                in_dim = initial_edge_repr_dim
                self.edge_weight_extractors[str(edge_type)] = ExtractorMLP(in_dim).to(device)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        x_dict['ligand'] = self.fastkan_ligand1(x_dict['ligand'])
        x_dict['ligand'] = self.fastkan_ligand2(x_dict['ligand'])
        x_dict['protein'] = self.fastkan_protein1(x_dict['protein'])
        x_dict['protein'] = self.fastkan_protein2(x_dict['protein'])
        
        ligand_graph_feat = self.global_pool_ligand(x_dict['ligand'], batch_dict['ligand'])
        protein_graph_feat = self.global_pool_protein(x_dict['protein'], batch_dict['protein'])
        combined_graph_feat = torch.cat([ligand_graph_feat, protein_graph_feat], dim=1)
        
        # 生成20×20图像特征
        img_feat = self.feat_to_img(combined_graph_feat)
        assert img_feat.shape[2] == 20 and img_feat.shape[3] == 20, \
            f"Swin输入尺寸错误：期望(3,20,20)，实际{img_feat.shape[1:4]}"
        
        swin_output = self.swin_transformer(img_feat)
        swin_output = self.swin_proj(swin_output)
        
        src, dst = edge_index_dict[('ligand', 'to', 'protein')]
        edge_repr = torch.cat([x_dict['ligand'][src], x_dict['protein'][dst]], dim=-1)
        d_pl = self.edge_lin(edge_attr_dict[('ligand', 'to', 'protein')])
        edge_repr = torch.cat((edge_repr, d_pl), dim=1)
        m_pl = self.edge_mlp(edge_repr)
        edge_batch = batch_dict['ligand'][src]
        
        w_pl = torch.tanh(self.lin_mpl(m_pl))
        m_w = w_pl * m_pl
        true_batch_size = ligand_graph_feat.shape[0]
        m_w = scatter_sum(m_w, edge_batch, dim=0, dim_size=true_batch_size)
        m_max, _ = scatter_max(m_pl, edge_batch, dim=0, dim_size=true_batch_size)
        m_out = torch.cat((m_w, m_max), dim=1)
        
        if m_out.shape[0] < true_batch_size:
            if not self.mean_initialized and self.training:
                valid_indices = torch.unique(edge_batch)
                if len(valid_indices) > 0:
                    self.m_out_mean = m_out[valid_indices].mean(dim=0)
                    self.mean_initialized = True
            pad_size = true_batch_size - m_out.shape[0]
            pad_tensor = self.m_out_mean.expand(pad_size, -1)
            m_out = torch.cat([m_out, pad_tensor], dim=0)
        
        edge_weights_dict = {}
        for edge_type in self.edge_types:
            src, dst = edge_index_dict.get(edge_type, (None, None))
            if src is None or dst is None:
                continue
            
            edge_repr = torch.cat([x_dict[edge_type[0]][src], x_dict[edge_type[2]][dst], edge_attr_dict[edge_type]], dim=-1)
            
            if str(edge_type) not in self.edge_weight_extractors:
                in_dim = edge_repr.shape[1]
                self.edge_weight_extractors[str(edge_type)] = ExtractorMLP(in_dim).to(edge_repr.device)
            
            edge_weights = self.edge_weight_extractors[str(edge_type)](edge_repr)
            edge_weights_dict[edge_type] = edge_weights
            edge_attr_dict[edge_type] = edge_attr_dict[edge_type] * torch.sigmoid(edge_weights)
        
        combined = torch.cat([m_out, swin_output], dim=1)
        return combined, edge_weights_dict


# 主网络（与训练一致）
class MainNet(torch.nn.Module):
    def __init__(self, metadata, initial_edge_repr_dim=None, img_size=20):
        super().__init__()
        hidden_channels = 64
        out_channels = 8
        edge_dim = 12
        
        self.ligand_att = MultiModalAttention(18, hidden_channels, out_channels, edge_dim)
        self.protein_att = MultiModalAttention(18, hidden_channels, out_channels, edge_dim)
        
        # 强制传递20×20尺寸到InterGNN
        self.heterognn = InterGNN(
            metadata, 
            edge_dim=edge_dim, 
            hidden_channels=hidden_channels, 
            out_channels=8, 
            num_layers=3,
            initial_edge_repr_dim=initial_edge_repr_dim,
            img_size=img_size  # 固定为20
        )
        
        self.protein_seq_mpl = MLP([480, 1024, 512, 16], dropout=0.1)
        self.lig_smiles_mpl = MLP([256, 512, 128, 16], dropout=0.3)
        
        # 输出层（与训练时保持一致）
        self.out = MLP([144, 256, 32, 8, 1], dropout=0.1)

    def forward(self, data):
        g_l, g_p, g_pl, pro_seq, lig_smiles = data
        
        l_att = self.ligand_att(g_l.x, g_l.edge_index, g_l.edge_attr, g_l.batch)
        p_att = self.protein_att(g_p.x, g_p.edge_index, g_p.edge_attr, g_p.batch)
        
        complex, edge_weights = self.heterognn(g_pl.x_dict, g_pl.edge_index_dict, g_pl.edge_attr_dict, g_pl.batch_dict)
        
        p_seq = self.protein_seq_mpl(pro_seq)
        l_smiles = self.lig_smiles_mpl(lig_smiles)
        
        # 拼接所有特征
        emb = torch.cat((l_att, p_att, complex, p_seq, l_smiles), dim=1)
        
        y_hat = self.out(emb).squeeze()
        
        return y_hat, edge_weights
    
# ====================== 3. 核心测试函数定义 ====================== #
def load_models(model_paths, metadata, edge_repr_dim, img_size=20):
    """加载多个训练好的模型并返回模型列表"""
    models = []
    for idx, path in enumerate(model_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 初始化模型
        model = MainNet(
            metadata=metadata,
            initial_edge_repr_dim=edge_repr_dim,
            img_size=img_size
        ).to(device)
        
        # 加载权重
        model.load_state_dict(torch.load(path, map_location=device), strict=True)
        model.eval()  # 设置为评估模式
        models.append(model)
        print(f"成功加载模型 {idx+1}/{len(model_paths)}: {path}")
    
    return models


def get_edge_repr_dim_from_data(test_loader, metadata, device, img_size=20):
    """从测试数据中获取边特征维度（与训练时保持一致的计算逻辑）"""
    # 初始化一个临时的InterGNN用于计算边特征维度
    temp_model = InterGNN(
        metadata=metadata,
        edge_dim=12,
        hidden_channels=64,
        out_channels=8,
        num_layers=3,
        img_size=img_size
    ).to(device)
    temp_model.eval()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            try:
                data = set_gpu(data, device)
                g_l, g_p, g_pl, pro_seq, lig_smiles = data
                
                x_dict = g_pl.x_dict
                edge_index_dict = g_pl.edge_index_dict
                edge_attr_dict = g_pl.edge_attr_dict
                
                # 执行KAN层计算（与训练时一致）
                x_dict['ligand'] = temp_model.fastkan_ligand1(x_dict['ligand'])
                x_dict['ligand'] = temp_model.fastkan_ligand2(x_dict['ligand'])
                x_dict['protein'] = temp_model.fastkan_protein1(x_dict['protein'])
                x_dict['protein'] = temp_model.fastkan_protein2(x_dict['protein'])
                
                # 计算第一个边类型的特征维度
                for edge_type in metadata[1]:
                    src, dst = edge_index_dict.get(edge_type, (None, None))
                    if src is None or dst is None:
                        continue
                    
                    edge_repr = torch.cat([
                        x_dict[edge_type[0]][src],
                        x_dict[edge_type[2]][dst],
                        edge_attr_dict[edge_type]
                    ], dim=-1)
                    return edge_repr.shape[1]  # 返回实际维度
            
            except Exception as e:
                print(f"计算边特征维度时出错（批次 {batch_idx}）: {e}")
                continue
    
    return 18  # 默认维度（与训练时保持一致）

# 新增的TXT日志写入函数
# 新增的TXT日志写入函数（确保datetime调用正确）
def write_txt_log(file_path, content, mode="a"):
    """
    写入内容到TXT文件，自动添加时间戳
    :param file_path: TXT文件路径
    :param content: 要写入的内容（字符串）
    :param mode: 写入模式（a=追加，w=覆盖）
    """
    # 直接使用已导入的datetime模块，调用其下的datetime类
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(file_path, mode, encoding="utf-8") as f:
        f.write(f"{timestamp} {content}\n")


# 调整后的单个数据集测试函数
# 调整后的单个数据集测试函数（输出每个模型的结果）
# 调整后的单个数据集测试函数（输出每个模型的结果）
def test_single_dataset(dataset_name, dataset_path, model_paths, output_dir, batch_size=16):
    """测试单个数据集并输出每个模型的单独结果及汇总结果"""
    print(f"\n===== 开始测试数据集: {dataset_name} =====")
    result_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    txt_log_path = os.path.join(result_dir, "test_results.txt")
    
    # 初始化日志（修正datetime调用：使用datetime.datetime.now()）
    write_txt_log(txt_log_path, f"测试数据集: {dataset_name}", "w")
    write_txt_log(txt_log_path, f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_txt_log(txt_log_path, f"数据集路径: {dataset_path}")

    # 1. 加载测试数据
    test_set = PLBA_Dataset('file', dataset_path)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    sample_count = len(test_set)
    write_txt_log(txt_log_path, f"测试样本数量: {sample_count}")
    print(f"测试样本数量: {sample_count}")

    # 2. 获取元数据和边特征维度
    sample_data = test_set[0]
    metadata = sample_data[2].metadata()
    edge_repr_dim = get_edge_repr_dim_from_data(test_loader, metadata, device)
    write_txt_log(txt_log_path, f"边特征维度: {edge_repr_dim}")

    # 3. 加载所有模型
    models = load_models(model_paths, metadata, edge_repr_dim)
    model_count = len(models)
    write_txt_log(txt_log_path, f"成功加载模型数量: {model_count}")

    # 4. 记录每个模型的预测结果和真实标签
    all_model_predictions = [[] for _ in range(model_count)]  # 每个模型的预测列表
    all_targets = []  # 真实标签

    with torch.no_grad():
        for batch in test_loader:
            data = set_gpu(batch, device)
            targets = data[0].y.cpu().numpy()
            all_targets.extend(targets)

            # 每个模型单独预测
            for i, model in enumerate(models):
                pred, _ = model(data)
                all_model_predictions[i].extend(pred.cpu().numpy())

    # 转换为numpy数组
    all_targets = np.array(all_targets)
    all_model_predictions = [np.array(preds) for preds in all_model_predictions]
    
    # 计算集成预测（平均所有模型）
    avg_predictions = np.mean(all_model_predictions, axis=0)

    # 5. 计算每个模型的指标和集成指标
    model_metrics = []
    for i in range(model_count):
        metrics = metrics_reg(all_targets, all_model_predictions[i])
        model_metrics.append(metrics)
    
    ensemble_metrics = metrics_reg(all_targets, avg_predictions)

    # 6. 输出并保存每个模型的结果
    write_txt_log(txt_log_path, "\n===== 各模型测试指标 =====")
    print("\n===== 各模型测试指标 =====")
    
    # 保存每个模型的详细指标到TXT
    for i in range(model_count):
        model_name = f"模型{i+1}"
        model_path = model_paths[i].split("/")[-1]  # 获取文件名
        write_txt_log(txt_log_path, f"\n{model_name} ({model_path}):")
        print(f"\n{model_name} ({model_path}):")
        
        for name, value in model_metrics[i].items():
            write_txt_log(txt_log_path, f"  {name}: {value}")
            print(f"  {name}: {value}")

    # 7. 输出并保存集成模型结果
    write_txt_log(txt_log_path, "\n===== 集成模型测试指标 =====")
    print("\n===== 集成模型测试指标 =====")
    for name, value in ensemble_metrics.items():
        write_txt_log(txt_log_path, f"{name}: {value}")
        print(f"{name}: {value}")

    # 8. 保存详细预测结果（CSV）
    detail_path = os.path.join(result_dir, "predictions_detail.csv")
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 表头：真实值 + 每个模型预测值 + 集成预测值
        headers = ["真实值"] + [f"模型{i+1}预测值" for i in range(model_count)] + ["集成预测值"]
        writer.writerow(headers)
        
        # 写入每行数据
        for idx in range(len(all_targets)):
            row = [all_targets[idx]]
            row.extend([all_model_predictions[i][idx] for i in range(model_count)])
            row.append(avg_predictions[idx])
            writer.writerow(row)

    # 9. 保存指标汇总（CSV）
    metric_path = os.path.join(result_dir, "metrics_summary.csv")
    with open(metric_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 表头
        headers = ["模型", "MAE", "RMSE", "Pearson R", "SD", "Spearman S", "CI"]
        writer.writerow(headers)
        
        # 写入每个模型的指标
        for i in range(model_count):
            model_name = f"模型{i+1}"
            metrics = model_metrics[i]
            writer.writerow([model_name, metrics["MAE"], metrics["RMSE"], 
                            metrics["Pearson R"], metrics["SD"], 
                            metrics["Spearman S"], metrics["CI"]])
        
        # 写入集成模型指标
        writer.writerow(["集成模型", ensemble_metrics["MAE"], ensemble_metrics["RMSE"],
                        ensemble_metrics["Pearson R"], ensemble_metrics["SD"],
                        ensemble_metrics["Spearman S"], ensemble_metrics["CI"]])

    write_txt_log(txt_log_path, f"\n结果保存路径: {result_dir}")
    print(f"结果已保存到: {result_dir}")
    return model_metrics, ensemble_metrics


# 修复后的批量测试函数（仅保留一个定义，修正datetime调用）
def batch_test_datasets(test_datasets, model_paths, output_root):
    """批量测试多个数据集，输出每个模型的单独结果"""
    overall_model_metrics = {}  # 每个数据集的各模型指标
    overall_ensemble_metrics = {}  # 每个数据集的集成指标
    summary_txt_path = os.path.join(output_root, "all_datasets_summary.txt")
    
    # 初始化汇总日志（修正datetime调用：使用datetime.datetime.now()）
    write_txt_log(summary_txt_path, "所有数据集测试汇总", "w")
    write_txt_log(summary_txt_path, f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_txt_log(summary_txt_path, f"模型数量: {len(model_paths)}")

    for dataset_name, dataset_path in test_datasets.items():
        # 正确接收元组返回值：(模型指标列表, 集成指标)
        model_metrics_list, ensemble_metrics = test_single_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            model_paths=model_paths,
            output_dir=output_root,
            batch_size=BATCH_SIZE
        )
        overall_model_metrics[dataset_name] = model_metrics_list
        overall_ensemble_metrics[dataset_name] = ensemble_metrics

    # 生成全局汇总报告
    write_txt_log(summary_txt_path, "\n===== 所有数据集指标汇总 =====")
    summary_csv_path = os.path.join(output_root, "all_datasets_summary.csv")
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["数据集", "模型", "MAE", "RMSE", "Pearson R", "SD", "Spearman S", "CI"]
        writer.writerow(headers)
        
        for dataset_name in test_datasets.keys():
            # 写入每个模型的指标
            for i, metrics in enumerate(overall_model_metrics[dataset_name]):
                writer.writerow([dataset_name, f"模型{i+1}", 
                                metrics["MAE"], metrics["RMSE"],
                                metrics["Pearson R"], metrics["SD"],
                                metrics["Spearman S"], metrics["CI"]])
            
            # 写入集成模型指标
            ens_metrics = overall_ensemble_metrics[dataset_name]
            writer.writerow([dataset_name, "集成模型",
                            ens_metrics["MAE"], ens_metrics["RMSE"],
                            ens_metrics["Pearson R"], ens_metrics["SD"],
                            ens_metrics["Spearman S"], ens_metrics["CI"]])

    # 修正测试完成时间的datetime调用
    write_txt_log(summary_txt_path, "\n" + "="*80)
    write_txt_log(summary_txt_path, f"所有数据集测试完成，完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_txt_log(summary_txt_path, f"汇总结果保存路径: {summary_csv_path}")
    write_txt_log(summary_txt_path, f"汇总日志路径: {summary_txt_path}")
    write_txt_log(summary_txt_path, "="*80)
    
    print(f"\n所有数据集汇总结果已保存到: {summary_csv_path}")
    print(f"所有数据集汇总日志已保存到: {summary_txt_path}")



# ====================== 4. 主函数入口 ====================== #
if __name__ == "__main__":
    # 检查模型路径是否存在
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}，请检查MODEL_PATHS配置")
    
    # 检查测试数据集是否存在
    for name, path in TEST_DATASETS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"测试数据集不存在: {path}，请检查TEST_DATASETS配置")
    
    # 执行批量测试
    batch_test_datasets(
        test_datasets=TEST_DATASETS,
        model_paths=MODEL_PATHS,
        output_root=OUTPUT_ROOT
    )
    
    print("\n===== 所有测试完成 =====")