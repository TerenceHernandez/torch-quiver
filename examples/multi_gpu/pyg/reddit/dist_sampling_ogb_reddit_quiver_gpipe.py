import os
from typing import Union, List

import torch
from torch import Tensor
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch_sparse import SparseTensor
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import ReLU, Dropout, LogSoftmax

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import SAGEConv, Sequential, GCNConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

import time

######################
# Import From Quiver
######################
import quiver


class SAGE(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels,
							 num_layers=2):
		super(SAGE, self).__init__()
		self.num_layers = num_layers

		self.convs = torch.nn.ModuleList()
		self.convs.append(SAGEConv(in_channels, hidden_channels))
		for _ in range(self.num_layers - 2):
			self.convs.append(SAGEConv(hidden_channels, hidden_channels))
		self.convs.append(SAGEConv(hidden_channels, out_channels))

	def forward(self, x, adjs):
		for i, (edge_index, _, size) in enumerate(adjs):
			x_target = x[:size[1]]  # Target nodes are always placed first.
			x = self.convs[i]((x, x_target), edge_index)
			if i != self.num_layers - 1:
				x = F.relu(x)
				x = F.dropout(x, p=0.5, training=self.training)
		return x.log_softmax(dim=-1)

	@torch.no_grad()
	def inference(self, x_all, device, subgraph_loader):
		pbar = tqdm(total=x_all.size(0) * self.num_layers)
		pbar.set_description('Evaluating')

		for i in range(self.num_layers):
			xs = []
			for batch_size, n_id, adj in subgraph_loader:
				edge_index, _, size = adj.to(device)
				x = x_all[n_id].to(device)
				x_target = x[:size[1]]
				x = self.convs[i]((x, x_target), edge_index)
				if i != self.num_layers - 1:
					x = F.relu(x)
				xs.append(x)

				pbar.update(batch_size)

			x_all = torch.cat(xs, dim=0)

		pbar.close()

		return x_all


class PipelineableSAGEConv(MessagePassing):
	
	def __init__(self, rank, layer, in_channels, out_channels, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.rank = rank

		self.conv = SAGEConv(in_channels, out_channels, *args, *kwargs)
		self.layer = layer

	def reset_parameters(self):
		self.conv.reset_parameters()

	def forward(self, x: Union[Tensor, OptPairTensor], adjs: List[Adj]) -> Tensor:
		# Calculate the right layers
		edge_index, _, size = adjs[self.layer]
		x_target = x[:size[:1]]

		if self.rank == 0:
			print(f'layer:{self.layer}', x.size())

		after_SAGE = self.conv((x, x_target), edge_index)
		if self.rank == 0:
			print(f'layer:{self.layer}, after_SAGE:', after_SAGE.size())

		return after_SAGE

	def message(self, x_j: Tensor) -> Tensor:
		return self.conv.message(x_j)

	def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
		return self.conv.message_and_aggregate(adj_t, x)


	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
															 self.out_channels)


# class SequentialGraphSAGE(torch.nn.Module):
# 	def __init__(self, in_channels, hidden_channels, out_channels,
# 							 num_layers=2):
# 		super(SequentialGraphSAGE, self).__init__()
#
# 		# self.convs.append(SAGEConv(in_channels, hidden_channels))
# 		# for _ in range(self.num_layers - 2):
# 		# 	self.convs.append(SAGEConv(hidden_channels, hidden_channels))
# 		# self.convs.append(SAGEConv(hidden_channels, out_channels))
#
# 		# def forward(self, x, adjs):
# 		# 	for i, (edge_index, _, size) in enumerate(adjs):
# 		# 		x_target = x[:size[1]]  # Target nodes are always placed first.
# 		# 		x = self.convs[i]((x, x_target), edge_index)
# 		# 		if i != self.num_layers - 1:
# 		# 			x = F.relu(x)
# 		# 			x = F.dropout(x, p=0.5, training=self.training)
# 		# 	return x.log_softmax(dim=-1)

def run(rank, world_size, data_split, edge_index, x, quiver_sampler, y, num_features, num_classes):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	dist.init_process_group('nccl', rank=rank, world_size=world_size)

	torch.torch.cuda.set_device(rank)

	train_mask, val_mask, test_mask = data_split
	train_idx = train_mask.nonzero(as_tuple=False).view(-1)
	train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

	train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, shuffle=True, drop_last=True)

	if rank == 0:
		subgraph_loader = NeighborSampler(edge_index, node_idx=None,
																			sizes=[-1], batch_size=2048,
																			shuffle=False, num_workers=6)

	torch.manual_seed(12345)

	# model = SAGE(num_features, 256, num_classes).to(rank)

	# Using Sequential Sage instead
	hidden_channels = 256
	model = Sequential(
		'x, adjs,', [
			(PipelineableSAGEConv(rank=rank, layer=0, in_channels=num_features, out_channels=hidden_channels), 'x, adjs -> x1a'),
			(ReLU(), 'x1a -> x1b'),
			(Dropout(p=0.5), 'x1b -> x1c'),
			(PipelineableSAGEConv(rank=rank, layer=1, in_channels=hidden_channels, out_channels=num_classes), 'x1c, adjs -> x2a'),
			(LogSoftmax(dim=-1), 'x2a -> x2b')
		]
	)
	model.to(rank)

	model = DistributedDataParallel(model, device_ids=[rank])
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	# Simulate cases those data can not be fully stored by GPU memory
	y = y.to(rank)

	for epoch in range(1, 21):
		model.train()
		epoch_start = time.time()
		for seeds in train_loader:
			n_id, batch_size, adjs = quiver_sampler.sample(seeds)
			adjs = [adj.to(rank) for adj in adjs]

			optimizer.zero_grad()
			out = model(x[n_id].to(rank), adjs)
			if rank == 0:
				print('OUT', out.size())
				print('BATCH_SIZE', batch_size)
			loss = F.nll_loss(out, y[n_id[:batch_size]])
			loss.backward()
			optimizer.step()

		dist.barrier()

		if rank == 0:
			print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {time.time() - epoch_start}')

		if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
			model.eval()
			with torch.no_grad():
				out = model.module.inference(x, rank, subgraph_loader)
			res = out.argmax(dim=-1) == y
			acc1 = int(res[train_mask].sum()) / int(train_mask.sum())
			acc2 = int(res[val_mask].sum()) / int(val_mask.sum())
			acc3 = int(res[test_mask].sum()) / int(test_mask.sum())
			print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

		dist.barrier()

	dist.destroy_process_group()


if __name__ == '__main__':
	dataset = Reddit('/data/Reddit')
	world_size = torch.cuda.device_count()

	data = dataset[0]

	csr_topo = quiver.CSRTopo(data.edge_index)

	##############################
	# Create Sampler And Feature
	##############################
	quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10], 0, mode='GPU')

	quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size="2G",
																	cache_policy="device_replicate", csr_topo=csr_topo)
	quiver_feature.from_cpu_tensor(data.x)

	print('Let\'s use', world_size, 'GPUs!')
	data_split = (data.train_mask, data.val_mask, data.test_mask)
	mp.spawn(
		run,
		args=(world_size, data_split, data.edge_index, quiver_feature, quiver_sampler, data.y, dataset.num_features,
					dataset.num_classes),
		nprocs=world_size,
		join=True
	)