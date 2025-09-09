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
from Swin_Transformer.model import swin_smaller_patch4_window5_1

# ====================== 参数设置（核心修改：输出路径结构）====================== #
k = 5
seed = 100
batch_size = 16
num_epochs = 70

# 图像尺寸配置
IMG_SIZE_LIST = [20]
# 根输出路径（用户指定）
ROOT_OUTPUT_DIR = "/opt/data/private/two/outputs20"
# 尺寸-子目录映射
SIZE_TO_SUBDIR = {
    20: "20graph"
}

# 设备设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------------------- 边类型感知的AttentiveFP模块 ---------------------- #
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


# ---------------------- 多模态注意力模块 ---------------------- #
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


# ---------------------- 交互图模块（仅适配20×20图像） ---------------------- #
class InterGNN(torch.nn.Module):
    def __init__(self, metadata, edge_dim, hidden_channels, out_channels, num_layers, initial_edge_repr_dim=None, img_size=20):
        super().__init__()
        self.edge_mlp = MLP([128 + 8, 512, 64, 16], dropout=0.1)
        self.lin_mpl = Linear(16, 16)
        self.edge_lin = Linear(1, 8)
        
        # 仅适配20×20的Swin Transformer
        self.img_size = img_size
        # 强制校验：仅允许20×20尺寸
        if self.img_size != 20:
            raise ValueError(f"Swin模块仅支持20×20图像，当前输入尺寸：{self.img_size}×{self.img_size}")
        
        self.swin_transformer = swin_smaller_patch4_window5_1(pretrained=False)
        # 20×20图像下采样后为5×5（20/4），适配窗口大小5×5
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
        self.target_dim = 20 * 20  # 20×20的总像素数
        self.feat_to_img = nn.Sequential(
            Linear(2 * hidden_channels, 256),
            nn.ReLU(),
            Linear(256, 3 * self.target_dim),  # 3通道（RGB），总维度=3×20×20=1200
            nn.Unflatten(1, (3, 20, 20))  # 强制输出(3,20,20)，与Swin输入匹配
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
        # 校验图像尺寸：确保输入Swin的是20×20
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
                print(f"动态创建边权重提取器 {edge_type}，输入维度: {in_dim}")
            
            edge_weights = self.edge_weight_extractors[str(edge_type)](edge_repr)
            edge_weights_dict[edge_type] = edge_weights
            edge_attr_dict[edge_type] = edge_attr_dict[edge_type] * torch.sigmoid(edge_weights)
        
        combined = torch.cat([m_out, swin_output], dim=1)
        # 新增：打印combined的维度，帮助验证输出层输入维度
        # print(f"combined维度: {combined.shape}")
        return combined, edge_weights_dict


# 主网络（仅传递20×20尺寸）
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
        
        # 核心修改：根据错误信息将输入维度从160调整为144
        # 原代码: self.out = MLP([16 + 16 + 96 + 16 + 16, 256, 32, 8, 1], dropout=0.1)
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
        
        # 新增：打印拼接后的维度，验证是否与输出层匹配
        # print(f"embedding维度: {emb.shape}")
        
        y_hat = self.out(emb).squeeze()
        
        return y_hat, edge_weights


# 数据集类
class PLBA_Dataset(Dataset):
    def __init__(self, *args):
        if args[0] == "file":
            filepath = args[1]
            f = open(filepath, 'rb')
            self.G_list = pickle.load(f)
            self.len = len(self.G_list)
        elif args[0] == 'list':
            self.G_list = args[1]
            self.len = len(args[1])

    def __getitem__(self, index):
        G = self.G_list[index]
        G[5] = torch.tensor(G[5], dtype=torch.float32)
        return G[0], G[1], G[2], G[3], G[5]

    def __len__(self):
        return self.len

    def k_fold(self, k, seed, batch_size):
        total_size = len(self.G_list)
        fold_len = int(total_size / k)
        indices = list(range(total_size))
        random.seed(seed)
        random.shuffle(indices)

        for select in range(k):
            print(f"Fold {select + 1}/{k}")
            if select == k - 1:
                val_idx = indices[select * fold_len:]
            else:
                val_idx = indices[select * fold_len:(select + 1) * fold_len]

            train_idx = list(set(indices) - set(val_idx))
            train_idx.sort(key=indices.index)

            train_list = [self.G_list[i] for i in train_idx]
            val_list = [self.G_list[i] for i in val_idx]

            train_set = PLBA_Dataset('list', train_list)
            val_set = PLBA_Dataset('list', val_list)
        
            if len(train_set) < batch_size:
                raise ValueError(f"Fold {select+1} 训练样本数不足（{len(train_set)} < {batch_size}), 请调整batch_size或数据集")
        
            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=False)
            val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)

            yield train_loader, val_loader


# 工具函数
def set_gpu(data, device):
    data_gpu = []
    for g in data:
        data_gpu.append(g.to(device))
    return data_gpu

def metrics_reg(targets, predicts):
    mae = metrics.mean_absolute_error(y_true=targets, y_pred=predicts)
    rmse = np.sqrt(np.mean((np.array(targets) - np.array(predicts)) ** 2))
    r = scipy.stats.mstats.pearsonr(targets, predicts)[0]
    s, _ = spearmanr(targets, predicts)

    x = [[item] for item in predicts]
    lr = LinearRegression()
    lr.fit(X=x, y=targets)
    y_ = lr.predict(x)
    sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5

    targets_arr = np.array(targets)
    predicts_arr = np.array(predicts)
    ci = 1 - (np.sum((targets_arr - predicts_arr) ** 2) / np.sum((targets_arr - np.mean(targets_arr)) ** 2))

    return [mae, rmse, r, sd, s, ci]

def my_val(model, val_loader, device):
    p_affinity = []
    y_affinity = []
    val_loss = 0.0
    n = 0

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            data = set_gpu(data, device)
            predict, _ = model(data)
            loss = F.mse_loss(predict, data[0].y)
            val_loss += loss.item()
            n += 1
            
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())

    torch.cuda.empty_cache()
    avg_val_loss = val_loss / n if n > 0 else 0.0
    return metrics_reg(targets=y_affinity, predicts=p_affinity), avg_val_loss


# 预计算边特征维度的函数
def get_edge_repr_dim(train_loader, metadata, device, img_size=20):
    """仅适配20×20图像的边特征维度计算"""
    model = InterGNN(metadata, edge_dim=12, hidden_channels=64, out_channels=8, num_layers=3, img_size=img_size)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            try:
                data = set_gpu(data, device)
                g_l, g_p, g_pl, pro_seq, lig_smiles = data
                
                x_dict = g_pl.x_dict
                edge_index_dict = g_pl.edge_index_dict
                edge_attr_dict = g_pl.edge_attr_dict
                
                x_dict['ligand'] = model.fastkan_ligand1(x_dict['ligand'])
                x_dict['ligand'] = model.fastkan_ligand2(x_dict['ligand'])
                x_dict['protein'] = model.fastkan_protein1(x_dict['protein'])
                x_dict['protein'] = model.fastkan_protein2(x_dict['protein'])
                
                for edge_type in metadata[1]:
                    src, dst = edge_index_dict.get(edge_type, (None, None))
                    if src is None or dst is None:
                        continue
                    
                    edge_repr = torch.cat([
                        x_dict[edge_type[0]][src],
                        x_dict[edge_type[2]][dst],
                        edge_attr_dict[edge_type]
                    ], dim=-1)
                    
                    actual_dim = edge_repr.shape[1]
                    print(f"边类型 {edge_type} 在批次 {batch_idx} 中的实际维度: {actual_dim}")
                    return actual_dim
            
            except Exception as e:
                print(f"计算边特征维度时出错（批次 {batch_idx}）: {e}")
                continue
    
    default_dim = 18
    print(f"警告：使用默认边特征维度 {default_dim}")
    return default_dim


# 训练函数（仅针对20×20图像）
def my_train(train_loader, val_loader, test_set, metadata, kf_filepath, fold_num, fastkan_params, img_size=20):
    print('start training (仅处理20×20图像)')
    os.makedirs(kf_filepath, exist_ok=True)
    print(f"当前模型保存目录：{kf_filepath}")
    
    edge_repr_dim = get_edge_repr_dim(train_loader, metadata, device, img_size)
    
    model = MainNet(metadata=metadata, initial_edge_repr_dim=edge_repr_dim, img_size=img_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=fastkan_params['lr'], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    model.train()

    train_loss_list = []
    val_loss_list = []
    best_rmse = float('inf')
    final_model_path = os.path.join(kf_filepath, f'best_model_fold_{fold_num}_img20.pt')
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(fastkan_params['epochs']):
        model.train()
        train_loss_epoch = 0.0
        n_train = 0
        
        for i, data in enumerate(train_loader):
            data = set_gpu(data, device)
            optimizer.zero_grad()
            
            out, edge_weights = model(data)
            loss = F.mse_loss(out, data[0].y)
            
            if edge_weights:
                reg_loss = 0
                for weights in edge_weights.values():
                    reg_loss += torch.var(weights) * 0.1
                loss += reg_loss
            
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            n_train += 1
            
            if i % 5 == 0:
                print(f'Epoch: {epoch+1}/{fastkan_params["epochs"]}, Batch: {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            torch.cuda.empty_cache()
        
        avg_train_loss = train_loss_epoch / n_train if n_train > 0 else 0.0
        train_loss_list.append(avg_train_loss)
        
        val_err, avg_val_loss = my_val(model, val_loader, device)
        val_loss_list.append(avg_val_loss)
        val_mae, val_rmse, val_r, val_sd, val_s, val_ci = val_err
        
        print(f'Epoch: {epoch+1}/{fastkan_params["epochs"]}')
        print(f'  训练损失: {avg_train_loss:.4f}')
        print(f'  验证损失: {avg_val_loss:.4f}')
        print(f'  验证指标: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R={val_r:.4f}, S={val_s:.4f}, CI={val_ci:.4f}')
        
        scheduler.step(val_rmse)
        
        temp_model_path = os.path.join(kf_filepath, f'temp_model_fold_{fold_num}_img20.pt')
        torch.save(model.state_dict(), temp_model_path)
        
        if val_rmse < best_rmse:
            print(f'********RMSE提升（{best_rmse:.4f} → {val_rmse:.4f}），保存最佳模型到 {final_model_path}*********')
            torch.save(model.state_dict(), final_model_path)
            best_rmse = val_rmse
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"早停计数器: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"验证RMSE连续{early_stop_patience}轮未下降，触发早停")
                if not os.path.exists(final_model_path):
                    print(f"警告：未找到最佳模型，使用最后一个临时模型作为最佳模型")
                    os.rename(temp_model_path, final_model_path)
                break

    p_affinity, y_affinity = my_test(test_set, metadata, final_model_path, kf_filepath, edge_repr_dim, img_size)

    csv_file_path = os.path.join(kf_filepath, f'all_fold_test_results_img20.csv')
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Fold', 'True', 'Pre'])
        for real, pred in zip(y_affinity, p_affinity):
            writer.writerow([f'fold {fold_num}', real, pred])
        
        fold_metrics = metrics_reg(y_affinity, p_affinity)
        metric_names = ['MAE', 'RMSE', 'Pearson R', 'SD', 'Spearman S', 'CI']
        metric_str = ';'.join([f'{name}: {value}' for name, value in zip(metric_names, fold_metrics)])
        writer.writerow([f'fold {fold_num} metrics', metric_str])
    
    return p_affinity

# 测试函数（仅针对20×20图像）
def my_test(test_set, metadata, model_file, kf_filepath, edge_repr_dim, img_size=20):
    p_affinity = []
    y_affinity = []

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"模型文件不存在：{model_file}，训练过程可能出错")

    model = MainNet(metadata=metadata, initial_edge_repr_dim=edge_repr_dim, img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_file), strict=True)
    model.eval()
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=True, num_workers=0)

    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            data = set_gpu(data, device)
            predict, edge_weights = model(data)
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())
            
            if i == 0:
                edge_weights_path = os.path.join(kf_filepath, f'test_edge_weights_fold_img20.pkl')
                with open(edge_weights_path, 'wb') as f:
                    pickle.dump(edge_weights, f)
        
                important_edges = {}
                for edge_type, weights in edge_weights.items():
                    important_indices = (weights > 0.7).nonzero(as_tuple=True)[0]
                    if len(important_indices) > 0:
                        important_edges[edge_type] = {
                            'indices': important_indices,
                            'weights': weights[important_indices]
                        }
        
                important_edges_path = os.path.join(kf_filepath, f'test_important_edges_fold_img20.pkl')
                with open(important_edges_path, 'wb') as f:
                    pickle.dump(important_edges, f)

    return p_affinity, y_affinity

# 主函数
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    
    fastkan_params = {
        'lr': 0.00088,
        'epochs': 150
    }
    
    dataset_configs = [
        {
            "train_path": "/opt/data/private/two/data/20111train.pkl",
            "test_path": "/opt/data/private/two/data/20111test.pkl"
        },
    ]
    
    # 仅处理20×20图像
    img_size = 20
    print(f"\n===== 开始测试图像尺寸: {img_size}x{img_size} =====")
    subdir_name = SIZE_TO_SUBDIR[img_size]
    output_dir = os.path.join(ROOT_OUTPUT_DIR, subdir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"当前尺寸结果将保存到: {output_dir}")
    
    for config in dataset_configs:
        train_path = config["train_path"]
        test_path = config["test_path"]
        
        dataset = PLBA_Dataset('file', train_path)
        test_set = PLBA_Dataset('file', test_path)
        
        all_test_predictions = []
        all_test_targets = None
        edge_repr_dim = None
        
        fold_model_paths = []
        for fold_num, (train_loader, val_loader) in enumerate(dataset.k_fold(k, seed, batch_size), start=1):
            metadata = train_loader.dataset[0][2].metadata()
            
            if edge_repr_dim is None:
                edge_repr_dim = get_edge_repr_dim(train_loader, metadata, device, img_size)
            
            final_model_path = os.path.join(output_dir, f'best_model_fold_{fold_num}_img20.pt')
            fold_model_paths.append(final_model_path)
            
            p_affinity = my_train(
                train_loader=train_loader,
                val_loader=val_loader,
                test_set=test_set,
                metadata=metadata,
                kf_filepath=output_dir,
                fold_num=fold_num,
                fastkan_params=fastkan_params,
                img_size=img_size
            )
            all_test_predictions.append(p_affinity)
            
            if all_test_targets is None:
                model_path = fold_model_paths[-1]
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件不存在：{model_path}，请检查训练逻辑")
                _, all_test_targets = my_test(
                    test_set=test_set,
                    metadata=metadata,
                    model_file=model_path,
                    kf_filepath=output_dir,
                    edge_repr_dim=edge_repr_dim,
                    img_size=img_size
                )
        
        avg_predictions = np.mean(all_test_predictions, axis=0)
        csv_file_path = os.path.join(output_dir, f'all_fold_test_results_img20.csv')
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for real, pred in zip(all_test_targets, avg_predictions):
                writer.writerow(['fold mean', real, pred])
            
            avg_metrics = metrics_reg(all_test_targets, avg_predictions)
            metric_names = ['MAE', 'RMSE', 'Pearson R', 'SD', 'Spearman S', 'CI']
            metric_str = ';'.join([f'{name}: {value}' for name, value in zip(metric_names, avg_metrics)])
            writer.writerow(['fold mean metrics', metric_str])
        
        print(f"数据集 {train_path} 处理完成（图像尺寸: {img_size}x{img_size}）")
    print("20×20图像测试完成！")
