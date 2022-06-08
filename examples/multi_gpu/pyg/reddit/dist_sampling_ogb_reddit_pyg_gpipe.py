import os
from typing import Union, List, Optional

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch_geometric.typing import OptPairTensor, Adj
from torch_sparse import SparseTensor
from torchgpipe import GPipe
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

import time


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
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.rank = rank

		self.conv = SAGEConv(in_channels, out_channels, *args, *kwargs)
		self.layer = layer

	def reset_parameters(self):
		self.conv.reset_parameters()

	def forward(self, x, edj0, edj1):
		if self.training:
			# x, edge_index = x_adjs

			# Calculate the right layers
			# edge_index, _, size = adjs[self.layer]
			edge_index = edj0 if self.layer == 1 else edj1

			x_target = x[:edge_index[1].size(0)]

			# if self.rank == 0:
			# 	print(f'layer:{self.layer}', x.size(), size)

			after_SAGE = self.conv((x, x_target), edge_index)
			# if self.rank == 0:
			# 	print(f'layer:{self.layer}, after_SAGE:', after_SAGE.size())

			return after_SAGE, edj0, edj1

	# else:
	# 	# Already in the format that we want
	# 	return self.conv(x, adjs)

	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
															 self.out_channels)

	def message(self, x_j: Tensor) -> Tensor:
		return self.conv.message(x_j)

	def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
		return self.conv.message_and_aggregate(adj_t, x)

	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
															 self.out_channels)


class ModifiedReLU(Module):
	def __init__(self, inplace: bool = False):
		super(ModifiedReLU, self).__init__()
		self.inplace = inplace

	def forward(self, x, edj0, edj1):
		# x, adjs = x_adjs

		return F.relu(x, inplace=self.inplace), edj0, edj1


class _DropoutNd(Module):
	__constants__ = ['p', 'inplace']
	p: float
	inplace: bool

	def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
		super(_DropoutNd, self).__init__()
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, "
											 "but got {}".format(p))
		self.p = p
		self.inplace = inplace

	def extra_repr(self) -> str:
		return 'p={}, inplace={}'.format(self.p, self.inplace)


class ModifiedDropOut(_DropoutNd):

	def forward(self, x, edj0, edj1):
		# x, adjs = x_adjs

		return F.dropout(x, self.p, self.training, self.inplace), edj0, edj1


class ModifiedLogMax(Module):
	__constants__ = ['dim']
	dim: Optional[int]

	def __init__(self, dim: Optional[int] = None) -> None:
		super(ModifiedLogMax, self).__init__()
		self.dim = dim

	def __setstate__(self, state):
		self.__dict__.update(state)
		if not hasattr(self, 'dim'):
			self.dim = None

	def forward(self, x, edj0, edj1):
		# x, adjs = x_adjs

		return F.log_softmax(x, self.dim, _stacklevel=5)

	def extra_repr(self):
		return 'dim={dim}'.format(dim=self.dim)


def run(rank, world_size, data_split, edge_index, x, y, num_features, num_classes):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	dist.init_process_group('nccl', rank=rank, world_size=world_size)

	train_mask, val_mask, test_mask = data_split
	train_idx = train_mask.nonzero(as_tuple=False).view(-1)
	train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

	train_loader = NeighborSampler(edge_index, node_idx=train_idx,
																 sizes=[25, 10], batch_size=1024,
																 shuffle=True, persistent_workers=True,
																 num_workers=os.cpu_count() // world_size)

	if rank == 0:
		subgraph_loader = NeighborSampler(edge_index, node_idx=None,
																			sizes=[-1], batch_size=2048,
																			shuffle=False, num_workers=6)

	torch.manual_seed(12345)
	# model = SAGE(num_features, 256, num_classes).to(rank)
	hidden_channels = 256
	model = Sequential(
		PipelineableSAGEConv(rank=rank, layer=0, in_channels=num_features, out_channels=hidden_channels),
		ModifiedReLU(),
		ModifiedDropOut(p=0.5),
		PipelineableSAGEConv(rank=rank, layer=1, in_channels=hidden_channels, out_channels=num_classes),
		ModifiedLogMax(dim=-1)
	)
	model.to(rank)

	if rank == 0:
		print(model)

	model = GPipe(model, balance=[1, 2, 2], chunks=1, checkpoint='never')

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	# Simulate cases those data can not be fully stored by GPU memory
	x, y = x, y.to(rank)

	for epoch in range(1, 21):
		model.train()
		epoch_start = time.time()
		for batch_size, n_id, adjs in train_loader:
			adjs = [adj.edge_index for adj in adjs]
			adjs = [adj.to(rank) for adj in adjs]

			# adjs = torch.stack(adjs).to(rank)

			optimizer.zero_grad()
			out = model((x[n_id].to(rank), adjs[0], adjs[1]))
			loss = F.nll_loss(out, y[n_id[:batch_size]])
			loss.backward()
			optimizer.step()

		dist.barrier()

		if rank == 0:
			print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Time: {time.time() - epoch_start}')

		# if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
		# 	model.eval()
		# 	with torch.no_grad():
		# 		out = model.module.inference(x, rank, subgraph_loader)
		# 	res = out.argmax(dim=-1) == y
		# 	acc1 = int(res[train_mask].sum()) / int(train_mask.sum())
		# 	acc2 = int(res[val_mask].sum()) / int(val_mask.sum())
		# 	acc3 = int(res[test_mask].sum()) / int(test_mask.sum())
		# 	print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

		dist.barrier()

	dist.destroy_process_group()


if __name__ == '__main__':
	dataset = Reddit('/data/Reddit')
	data = dataset[0]
	world_size = 1  # torch.cuda.device_count()
	print('Let\'s use', world_size, 'GPUs!')
	data_split = (data.train_mask, data.val_mask, data.test_mask)
	mp.spawn(run,
					 args=(world_size, data_split, data.edge_index, data.x, data.y, dataset.num_features, dataset.num_classes),
					 nprocs=world_size, join=True)
