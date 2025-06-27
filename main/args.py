import argparse




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data params 数据参数---
    # 数据集
    parser.add_argument("--dataset", type=str.upper, default="smd")
    # 训练机
    parser.add_argument("--group", type=str, default="3-3", help="Required for SMD dataset. <group_index>-<index>")
    # 回溯窗口，可能类似滑动窗口
    parser.add_argument("--lookback", type=int, default=50)
    # 是否进行归一化
    parser.add_argument("--normalize", type=str2bool, default=True)
    # 它可能代表某个特定的结果或特殊的响应
    parser.add_argument("--spec_res", type=str2bool, default=False)





    # -- Model params 模型参数 ---s
    # 1D conv layer   卷积核为7的一维卷积层
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers  图注意力层 有标准GAT也可GATv2
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    # 特征图注意力的嵌入维度
    parser.add_argument("--feat_gat_embed_dim", type=int, default=0)
    # 时间图注意力的嵌入维度
    parser.add_argument("--time_gat_embed_dim", type=int, default=0)
    # GRU layer  可能是GRU网络层数
    parser.add_argument("--gru_n_layers", type=int, default=1)
    # GRU隐藏层的维度
    parser.add_argument("--gru_hid_dim", type=int, default=150)
    # Forecasting Model    全连接预测模型
    # 全连接层数
    parser.add_argument("--fc_n_layers", type=int, default=1)
    # 全连接隐藏层维度
    parser.add_argument("--fc_hid_dim", type=int, default=150)
    # Reconstruction Model  VAE重构模型
    # VAE层数
    parser.add_argument("--recon_n_layers", type=int, default=1)
    # VAE隐藏层维度
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)



    # --- Train params  训练参数---
    # 迭代次数
    parser.add_argument("--epochs", type=int, default=10)
    # 应该是按比例划分数据
    parser.add_argument("--val_split", type=float, default=0.1)
    # batch size值
    parser.add_argument("--bs", type=int, default=128)
    # 初始学习率
    parser.add_argument("--init_lr", type=float, default=1e-3)
    # 是否打乱数据顺序
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    # 是一种常用的正则化技术，用于减少神经网络模型的过拟合
    parser.add_argument("--dropout", type=float, default=0.3)
    # 是否用显卡
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    # 多少epoch打印一次信息或输出
    parser.add_argument("--print_every", type=int, default=1)
    # 是否将数据记录到可视化工具 TensorBoard 中
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)





    # --- Predictor params 异常预测器参数---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    # 动态阈值。
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    # --- Other ---
    parser.add_argument("--comment", type=str, default="")

    return parser
