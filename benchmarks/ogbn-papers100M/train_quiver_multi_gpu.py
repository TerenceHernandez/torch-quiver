import os
import time
import glob
import argparse
import os.path as osp

import numpy as np
from torch.utils import data
from tqdm import tqdm

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

import quiver
from quiver.feature import DeviceConfig, Feature
import gc

ROOT = '/data/terencehernandez'
CPU_CACHE_GB = 40
GPU_CACHE_GB = 20


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


class Paper100MDataset:
  def __init__(self, root, gpu_portion):
    data_dir = osp.join(root, 'ogbn_papers100M')
    feat_root = osp.join(data_dir, 'feat', 'sort_feature.pt')
    prev_root = osp.join(data_dir, 'feat', 'prev_order.pt')
    indptr_root = osp.join(data_dir, 'csr', 'indptr.pt')
    indices_root = osp.join(data_dir, 'csr', 'indices.pt')
    label_root = osp.join(data_dir, 'label', 'label.pt')
    index_root = osp.join(data_dir, 'index', 'train_idx.pt')
    feat = torch.load(feat_root)
    print('load feature')
    prev_order = torch.load(prev_root)
    node_count = prev_order.shape[0]
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(int(node_count * gpu_portion))
    new_order = torch.zeros_like(prev_order)
    prev_order[:int(node_count * gpu_portion)] = prev_order[perm_range]
    new_order[prev_order] = total_range
    feature = feat[prev_order]
    print('reorder feature')
    del feat
    self.feature = feature.share_memory_()
    self.indptr = torch.load(indptr_root).share_memory_()
    self.indices = torch.load(indices_root).share_memory_()
    self.label = torch.load(label_root).squeeze().share_memory_()
    self.train_idx = torch.load(index_root).share_memory_()
    self.new_order = new_order
    self.prev_order = prev_order


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all



class GNN(torch.nn.Module):
    def __init__(self,
                 model: str,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.model = model.lower()
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'gat':
            self.convs.append(
                GATConv(in_channels, hidden_channels // heads, heads))
            self.skips.append(Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels // heads, heads))
                self.skips.append(Linear(hidden_channels, hidden_channels))

        elif self.model == 'graphsage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward_step(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, (edge_index, _, size) in enumerate(adjs_t):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if self.model == 'gat':
                x = x + self.skips[i](x_target)
                x = F.elu(self.norms[i](x))
            elif self.model == 'graphsage':
                x = F.relu(self.norms[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def forward(self, batch, batch_idx: int):
        y_hat = self.forward_step(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        # self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
        #          on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc',
                 self.val_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc',
                 self.test_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


def run(rank, args, world_size, quiver_sampler, quiver_feature, label,
        train_idx, num_features, num_classes):

    torch.cuda.set_device(rank)
    print(f'{rank} beg')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    torch.manual_seed(123 + 45 * rank)
    #
    # gpu_size = GPU_CACHE_GB * 1024 * 1024 * 1024 // (768 * 4)
    # cpu_size = CPU_CACHE_GB * 1024 * 1024 * 1024 // (768 * 4)

    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=1024,
                                               pin_memory=True,
                                               shuffle=True)

    model = GNN(args.model,
                in_channels=num_features,
                out_channels=num_classes,
                hidden_channels=args.hidden_channels,
                num_layers=len(args.sizes),
                dropout=args.dropout).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # global2host = torch.load(osp.join(ROOT, f'{host_size}h/global2host.pt'))
    # replicate = torch.load(osp.join(ROOT, f'{host_size}h/replicate{host}.pt'))
    # info = quiver.feature.PartitionInfo(rank, host=host, hosts=host_size, global2host=global2host,
    #                                     replicate=replicate)
    #
    # prev_order = torch.load(
    #     osp.join('/data/mag/mag240m_kddcup2021', 'processed', 'paper',
    #              'prev_order2.pt'))
    # disk_map = torch.zeros(prev_order.size(0), device=rank,
    #                        dtype=torch.int64) - 1
    # mem_range = torch.arange(end=cpu_size + 2 * gpu_size,
    #                          device=rank,
    #                          dtype=torch.int64)
    # disk_map[prev_order[:2 * gpu_size + cpu_size]] = mem_range
    # print(f'{rank} disk map')
    #


    # quiver_feature.set_mmap_file(
    #     osp.join('/data/mag/mag240m_kddcup2021', 'processed', 'paper',
    #              'node_feat.npy'), disk_map)
    # print(f'{rank} mmap file')

    sample_time = []
    feat_time = []
    train_time = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_sample_time = []
        epoch_feat_time = []
        epoch_train_time = []

        epoch_beg = time.time()
        for cnt, seeds in enumerate(train_loader):
            t0 = time.time()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            t1 = time.time()
            x = quiver_feature[n_id]
            y = label[n_id[:batch_size]].to(torch.long)
            batch = Batch(x=x, y=y, adjs_t=adjs).to(rank)
            t2 = time.time()
            optimizer.zero_grad()
            loss = model(batch, 0)
            loss.backward()
            optimizer.step()
            t3 = time.time()
            epoch_sample_time.append(t1 - t0)
            epoch_feat_time.append(t2 - t1)
            epoch_train_time.append(t3 - t2)

        dist.barrier()
        # TODO could this be why it takes so long? -> Synchronisation

        if rank == 0:
            # remove 10% minium values and 10% maximum values
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {time.time() - epoch_beg}',
            )
            print(
                f'---- Sampling time: {np.sum(epoch_sample_time)},'
                f' Feat Aggregation time: {np.sum(epoch_feat_time)},'
                f' Train time: {np.sum(epoch_train_time)}.'
            )

        # if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
        #     model.eval()
        #     with torch.no_grad():
        #         out = model.module.inference(quiver_feature, rank, subgraph_loader)
        #     res = out.argmax(dim=-1) == y.cpu()
        #     acc1 = int(res[train_idx].sum()) / train_idx.numel()
        #     acc2 = int(res[val_idx].sum()) / val_idx.numel()
        #     acc3 = int(res[test_idx].sum()) / test_idx.numel()
        #     print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        # Average out epoch benchmark times
        sample_time.append(
          (np.sum(epoch_sample_time), np.min(epoch_sample_time), np.max(epoch_sample_time)))
        feat_time.append(
          (np.sum(epoch_feat_time), np.min(epoch_feat_time), np.max(epoch_feat_time)))
        train_time.append(
          (np.sum(epoch_train_time), np.min(epoch_train_time), np.max(epoch_train_time)))

        dist.barrier()

    if rank == 0:
        print("Sample time statistics:", sample_time)
        print("Feature aggregation time statistics:", feat_time)
        print("Trainiing time statistics:", train_time)

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model',
                        type=str,
                        default='graphsage',
                        choices=['gat', 'graphsage'])
    parser.add_argument('--sizes', type=str, default='15-10')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--root', type=str, default=ROOT)
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    world_size = args.num_gpus

    seed_everything(42)
    dataset = Paper100MDataset(args.root, 0.15 * min(world_size, 4))

    ##############################
    # Create Sampler And Feature
    ##############################
    csr_topo = quiver.CSRTopo(indptr=dataset.indptr, indices=dataset.indices)
    csr_topo.feature_order = dataset.new_order
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, args.sizes,
                                                 0,
                                                 mode="UVA")
    quiver_feature = quiver.Feature(rank=0,
                                    device_list=list(range(world_size)),
                                    device_cache_size="8G",
                                    cache_policy="p2p_clique_replicate",
                                    csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(dataset.feature)
    l = list(range(world_size))
    quiver.init_p2p(l)
    del dataset.feature

    print('Let\'s use', world_size, 'GPUs!')

    mp.spawn(run,
             args=(args, world_size, quiver_sampler, quiver_feature, dataset.label,
                   dataset.train_idx, 128, 172),
             nprocs=world_size,
             join=True)
