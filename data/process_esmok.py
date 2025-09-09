import os
import sys
import openbabel
from openbabel import pybel

# 设置BABEL_LIBDIR环境变量
# 根据你的OpenBabel安装位置调整路径
# 常见路径示例：
if sys.platform == 'linux':
    # Linux系统示例路径
    os.environ['BABEL_LIBDIR'] = '/root/miniconda3/envs/gign/lib/openbabel'
elif sys.platform == 'darwin':
    # macOS系统示例路径
    os.environ['BABEL_LIBDIR'] = '/Applications/OpenBabel/lib/openbabel'
else:
    # Windows系统示例路径
    os.environ['BABEL_LIBDIR'] = 'C:\\Program Files\\OpenBabel\\lib\\openbabel'


import sys
sys.path.insert(0, '/opt/data/private/two/esm')  # 添加本地库路径

# 直接导入预训练模型函数
from esm.pretrained import esm2_t12_35M_UR50D  # 关键修改
import torch
import openbabel
from openbabel import pybel
import warnings
warnings.filterwarnings('ignore')
import os
from torch_geometric.data import Data,HeteroData
from torch_geometric.utils import contains_isolated_nodes, tree_decomposition
from scipy.spatial import distance_matrix
import torch_geometric.transforms as T
import pickle
from openbabel_featurizer import Featurizer, CusBondFeaturizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline, BertTokenizer, BertModel,RobertaTokenizer, RobertaModel
import re
from prody import *
import networkx as nx
import numpy as np


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_grad_enabled(False) 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # if torch.cuda.is_available() else 'cpu'

# ---------------------- ESM-2 模型加载 ---------------------- #
def load_esm2_model(model_location="esm2_t12_35M_UR50D"):
    """从本地加载ESM-2模型并创建投影层"""
    # 直接导入底层函数
    from esm.pretrained import load_model_and_alphabet_hub
    model, alphabet = load_model_and_alphabet_hub(model_location)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    
    # 添加MLP投影层：480维 -> 768维
    projection = torch.nn.Linear(480, 480).to(device)
    projection.eval()  # 设置为评估模式
    return model, batch_converter, projection

# ---------------------- ChemBERTa 模型加载 ---------------------- #
def load_chemberta_model(local_path):
    """从本地路径加载ChemBERTa模型（基于RoBERTa）"""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"ChemBERTa模型路径不存在: {local_path}")
    
    try:
        tokenizer = RobertaTokenizer.from_pretrained(local_path, do_lower_case=False)
        model = RobertaModel.from_pretrained(local_path).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        raise ValueError(f"加载ChemBERTa模型失败: {e}")

def process_smiles(smiles, tokenizer, model, device):
    """处理SMILES字符串并获取嵌入"""
    if not smiles:
        return np.zeros(256, dtype=np.float32)
    
    with torch.no_grad():
        inputs = tokenizer(
            smiles, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=256
        ).to(device)
        
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 确保投影层在正确的设备上
        projection = torch.nn.Linear(768, 256).to(device)
        projected_embedding = projection(cls_embedding).squeeze().cpu().numpy()
        
        return projected_embedding


def info_3D(a, b, c):
    ab = b - a
    ac = c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_

def info_3D_cal(edge, ligand,h_num):
    node1_idx = edge[0]
    node2_idx = edge[1]
    atom1 = ligand.atoms[node1_idx]
    atom2 = ligand.atoms[node2_idx]

    neighbour1 = []
    neighbour2 = []
    for neighbour_atom in openbabel.OBAtomAtomIter(atom1.OBAtom):
        if neighbour_atom.GetAtomicNum() != 1:
            neighbour1.append(neighbour_atom.GetIdx() -h_num[neighbour_atom.GetIdx()] - 1)

    for neighbour_atom in openbabel.OBAtomAtomIter(atom2.OBAtom):
        if neighbour_atom.GetAtomicNum() != 1:
            neighbour2.append(neighbour_atom.GetIdx() -h_num[neighbour_atom.GetIdx()] - 1)

    neighbour1.remove(node2_idx)
    neighbour2.remove(node1_idx)
    neighbour1.extend(neighbour2)

    angel_list = []
    area_list = []
    distence_list = []

    if len(neighbour1) == 0 and len(neighbour2) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for node3_idx in neighbour1:
        node1_coord = np.array(ligand.atoms[node1_idx].coords)
        node2_coord = np.array(ligand.atoms[node2_idx].coords)
        node3_coord = np.array(ligand.atoms[node3_idx].coords)

        angel, area, distence = info_3D(node1_coord, node2_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    for node3_idx in neighbour2:
        node1_coord = np.array(ligand.atoms[node1_idx].coords)
        node2_coord = np.array(ligand.atoms[node2_idx].coords)
        node3_coord = np.array(ligand.atoms[node3_idx].coords)
        angel, area, distence = info_3D(node2_coord, node1_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    return [np.max(angel_list) * 0.01, np.sum(angel_list) * 0.01, np.mean(angel_list) * 0.01,
            np.max(area_list), np.sum(area_list), np.mean(area_list),
            np.max(distence_list) * 0.1, np.sum(distence_list) * 0.1, np.mean(distence_list) * 0.1]


def get_complex_edge_fea(edge_list,coord_list):

    net = nx.Graph()
    net.add_weighted_edges_from(edge_list)
    edges_fea = []
    for edge in edge_list:
        edge_fea = []

        # fea_angle = get_angleinfo(edge,net,coord_list)
        # edge_fea.extend(fea_3d)

        edge_fea.append(edge[2])
        edges_fea.append(edge_fea)

    return edges_fea

def read_ligand(filepath):
    featurizer = Featurizer(save_molecule_codes=False)
    ligand = next(pybel.readfile("mol2", filepath))
    ligand_coord, atom_fea,h_num = featurizer.get_features(ligand)

    return ligand_coord, atom_fea, ligand,h_num

def read_protein(filepath, esm2_model, batch_converter, projection):
    featurizer = Featurizer(save_molecule_codes=False)
    protein_pocket = next(pybel.readfile("pdb", filepath))
    pocket_coord, atom_fea,h_num = featurizer.get_features(protein_pocket)

    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}
    seq = ''
    protein_filepath = filepath.replace('pocket','protein')
    for line in open(protein_filepath):
        if line[0:6] == "SEQRES":
            columns = line.split()
            for resname in columns[4:]:
                if resname in aa_codes:
                    seq = seq + aa_codes[resname] + ' '
    sequences_Example = re.sub(r"[UZOB]", "X", seq)

    # 使用ESM-2提取特征并投影到768维
    with torch.no_grad():
        batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequences_Example)])
        batch_tokens = batch_tokens.to(device)
        results = esm2_model(batch_tokens, repr_layers=[12])  # 提取最后一层隐藏状态
        token_embeddings = results["representations"][12]  # [batch_size, seq_len, 480]
        seq_embedding = torch.mean(token_embeddings[0, 1:-1, :], dim=0, keepdim=True)  # [1, 480]
        pro_seq_emb = projection(seq_embedding).squeeze()  # [768]

    return pocket_coord, atom_fea,protein_pocket,h_num,pro_seq_emb

def Mult_graph(lig_file_name, pocket_file_name, id, score, esm2_model, batch_converter, projection, chemberta_tokenizer, chemberta_model):
    lig_coord, lig_atom_fea, mol, h_num_lig = read_ligand(lig_file_name)
    pocket_coord, pocket_atom_fea, protein, h_num_pro, pro_seq = read_protein(pocket_file_name, esm2_model, batch_converter, projection)
    
    # 处理SMILES获取嵌入
    smiles = mol.write("smi").strip() if mol else ""
    lig_smiles_emb = process_smiles(
        smiles, 
        chemberta_tokenizer, 
        chemberta_model, 
        device=device  # 显式传递设备
    ) if smiles else np.zeros(256, dtype=np.float32)
    
    if mol and protein:
        G_l = Ligand_graph(lig_atom_fea, mol, h_num_lig, score)
        G_p = protein_graph(pocket_atom_fea, protein, h_num_pro, score)
        G_inter = Inter_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, score)
        return [G_l, G_p, G_inter, pro_seq, id, lig_smiles_emb]
    else:
        return None

def bond_fea(bond,atom1,atom2):
    is_Aromatic = int(bond.IsAromatic())
    is_inring = int(bond.IsInRing())
    d = atom1.GetDistance(atom2)

    node1_idx = atom1.GetIdx()
    node2_idx = atom2.GetIdx()

    neighbour1 = []
    neighbour2 = []
    for neighbour_atom in openbabel.OBAtomAtomIter(atom1):
        if (neighbour_atom.GetAtomicNum() != 1 ) and (neighbour_atom.GetIdx() != node2_idx) :
            neighbour1.append(neighbour_atom)

    for neighbour_atom in openbabel.OBAtomAtomIter(atom2):
        if ( neighbour_atom.GetAtomicNum() != 1) and (neighbour_atom.GetIdx() != node1_idx):
            neighbour2.append(neighbour_atom)

    if len(neighbour1) == 0 and len(neighbour2) == 0:
        return [d,0, 0, 0, 0, 0, 0, 0, 0, 0,is_Aromatic,is_Aromatic]

    angel_list = []
    area_list = []
    distence_list = []

    node1_coord = np.array([atom1.GetX(),atom1.GetY(),atom1.GetZ()])
    node2_coord = np.array([atom2.GetX(),atom2.GetY(),atom2.GetZ()])

    for atom3 in neighbour1:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node1_coord, node2_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    for atom3 in neighbour2:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node2_coord, node1_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    return [d,
        np.max(angel_list) * 0.01, np.sum(angel_list) * 0.01, np.mean(angel_list) * 0.01,
        np.max(area_list), np.sum(area_list), np.mean(area_list),
        np.max(distence_list) * 0.1, np.sum(distence_list) * 0.1, np.mean(distence_list) * 0.1,
        is_Aromatic, is_inring]

def edgelist_to_tensor(edge_list):
    row = []
    column = []
    coo = []
    for edge in edge_list:
        row.append(edge[0])
        column.append(edge[1])

    coo.append(row)
    coo.append(column)

    coo = torch.Tensor(coo)
    edge_tensor = torch.tensor(coo, dtype=torch.long)
    return edge_tensor

def atomlist_to_tensor(atom_list):
    new_list = []
    for atom in atom_list:
        new_list.append([atom])
    atom_tensor = torch.Tensor(new_list)
    return atom_tensor

def Ligand_graph(lig_atoms_fea,ligand,h_num,score):
    edges = []
    edges_fea = []
    for bond in openbabel.OBMolBondIter(ligand.OBMol):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 1) or (atom2.GetAtomicNum() == 1):
            continue
        else:
            idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx()-1] - 1
            idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx()-1] - 1

            edge_fea = bond_fea(bond, atom1, atom2)
            edge = [idx_1, idx_2]
            edges.append(edge)
            edges_fea.append(edge_fea)

            re_edge = [idx_2, idx_1]
            edges.append(re_edge)
            edges_fea.append(edge_fea)

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    x = torch.tensor(lig_atoms_fea, dtype=torch.float32)
    edge_index = edgelist_to_tensor(edges)
    G_lig = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=torch.tensor(score))

    return G_lig

def protein_graph(pocket_atom_fea, protein,h_num,score):
    edges = []
    edges_fea = []
    for bond in openbabel.OBMolBondIter(protein.OBMol):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 1) or (atom2.GetAtomicNum() == 1):
            continue
        else:
            idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx()-1] -1
            idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx()-1] -1

            edge_fea = bond_fea(bond,atom1,atom2)
            edge = [idx_1,idx_2]
            edges.append(edge)
            edges_fea.append(edge_fea)

            re_edge = [idx_2,idx_1]
            edges.append(re_edge)
            edges_fea.append(edge_fea)

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    x = torch.tensor(pocket_atom_fea, dtype=torch.float32)
    edge_index = edgelist_to_tensor(edges)
    G_pocket = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(score))

    return G_pocket

def Inter_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, score, cut=5):
    coord_list = []
    for atom in lig_coord:
        coord_list.append(atom)
    for atom in pocket_coord:
        coord_list.append(atom)

    dis = distance_matrix(x=coord_list, y=coord_list)
    lenth = len(coord_list)
    edge_list = []

    edge_list_fea = []
    # Bipartite Graph; i belongs to ligand, j belongs to protein
    for i in range(len(lig_coord)):
        for j in range(len(lig_coord), lenth):
            if dis[i, j] < cut:
                edge_list.append([i, j-len(lig_coord), dis[i, j]])
                edge_list_fea.append([i,j,dis[i,j]])

    data = HeteroData()
    edge_index = edgelist_to_tensor(edge_list)

    data['ligand'].x = torch.tensor(lig_atom_fea, dtype=torch.float32)
    data['ligand'].y = torch.tensor(score)
    data['protein'].x = torch.tensor(pocket_atom_fea, dtype=torch.float32)
    data['ligand', 'protein'].edge_index = edge_index

    complex_edges_fea = get_complex_edge_fea(edge_list_fea,coord_list)
    edge_attr = torch.tensor(complex_edges_fea, dtype=torch.float32)
    data['ligand', 'protein'].edge_attr = edge_attr
    data = T.ToUndirected()(data)

    return data

def get_Resfea(res):
    aa_codes = {
        'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4,
        'PHE': 5, 'GLY': 6, 'HIS': 7, 'LYS': 8,
        'ILE': 9, 'LEU': 10, 'MET': 11, 'ASN': 12,
        'PRO': 13, 'GLN': 14, 'ARG': 15, 'SER': 16,
        'THR': 17, 'VAL': 18, 'TYR': 19, 'TRP': 0}
    one_hot = np.eye(21)
    if res in aa_codes:
        code = aa_codes[res]
    else:
        code = 20
    fea = one_hot[code]
    return fea

def GetPDBDict(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = {}
    for line in lines:
        if "//" in line:
            temp = line.split()
            name, score = temp[0], float(temp[3])
            res[name] = score
    return res

# ---------------------- 模型加载主逻辑 ---------------------- #
def process_raw_data(dataset_path, processed_file, affinity_txt_path="/data/pdb_affinity.txt"):
    # 加载PDB结合数据
    pdb_dict_path = "/opt/data/private/two/data/index/INDEX_general_PL_data.2020"
    if not os.path.exists(pdb_dict_path):
        raise FileNotFoundError(f"PDBDict文件不存在: {pdb_dict_path}")
    res = GetPDBDict(Path=pdb_dict_path)
    
    set_list = [x for x in os.listdir(dataset_path) if len(x) == 4]
    G_list = []

    # 打开文件用于保存PDBID和亲和力
    with open(affinity_txt_path, 'w') as f_affinity:
        # 写入表头
        f_affinity.write("PDBID\tAffinity\n")
        
        # ---------------------- 加载ESM-2（本地路径） ---------------------- #
        model_location = "esm2_t12_35M_UR50D"  # 替换为你的ESM-2模型路径
        esm2_model, batch_converter, projection = load_esm2_model(model_location)

        # ---------------------- 加载ChemBERTa（本地路径） ---------------------- #
        chemberta_path = "/ChemBERTa/ChemBERTa-zinc-base-v1"
        try:
            chemberta_tokenizer, chemberta_model = load_chemberta_model(chemberta_path)
        except Exception as e:
            raise ValueError(f"加载ChemBERTa失败: {e}")

        # ---------------------- 处理数据集 ---------------------- #
        for item in tqdm(set_list):
            try:
                score = res[item]
                # 将PDBID和亲和力写入文件
                f_affinity.write(f"{item}\t{score}\n")
                
                lig_file_name = os.path.join(dataset_path, item, f"{item}_ligand.mol2")
                pocket_file_name = os.path.join(dataset_path, item, f"{item}_pocket.pdb")
                
                G = Mult_graph(
                    lig_file_name, 
                    pocket_file_name, 
                    item, 
                    score, 
                    esm2_model, 
                    batch_converter,
                    projection,  # 添加投影层
                    chemberta_tokenizer, 
                    chemberta_model
                )
                
                if G:
                    G_list.append(G)
            except Exception as e:
                print(f"处理{item}时出错: {e}")
                continue

    print(f"成功处理样本数: {len(G_list)}")
    print(f"PDBID和亲和力已保存至: {affinity_txt_path}")
    with open(processed_file, 'wb') as f:
        pickle.dump(G_list, f)
    f.close()


def GetPDBDict(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = {}
    for line in lines:
        if "//" in line:
            temp = line.split()
            name, score = temp[0], float(temp[3])
            res[name] = score
    return res



if __name__ == "__main__":
    raw_data_path = "/data/higntest"
    data_path = "/data/Hightest.pkl"
    
    process_raw_data(raw_data_path, data_path)

