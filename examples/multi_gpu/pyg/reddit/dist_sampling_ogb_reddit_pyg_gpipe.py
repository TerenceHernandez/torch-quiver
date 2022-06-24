import argparse
import os
from typing import Union, List, Optional

import torch
from torch import Tensor
from torch.distributed.pipeline.sync import Pipe
from torch_geometric.utils import subgraph
from torchgpipe import GPipe
from torch.nn import Module, Sequential
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.typing import OptPairTensor, Adj
from torch_sparse import SparseTensor
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.distributed.rpc import init_rpc

from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

import time


class SAGE(torch.nn.Module):
	def __init__(self, x, in_channels, hidden_channels, out_channels,
							 num_layers=2):
		super(SAGE, self).__init__()
		self.num_layers = num_layers
		self.x = x

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


def evaluate(model: Sequential, x_all, device, subgraph_loader):
	convs = [model[0], model[3]]
	num_convs = len(convs)

	pbar = tqdm(total=x_all.size(0) * num_convs)
	pbar.set_description('Evaluating')

	for i in convs:
		xs = []
		for batch_size, n_id, adj in subgraph_loader:
			edge_index, _, size = adj.to(device)
			x = x_all[n_id].to(device)
			x_target = x[:size[1]]
			x = convs[i]((x, x_target), edge_index)
			if i != num_convs - 1:
				x = F.relu(x)
			xs.append(x)

			pbar.update(batch_size)

		x_all = torch.cat(xs, dim=0)

	pbar.close()

	return x_all


def print_device(data, text_to_print):
	try:
		d = data.get_device()
		print(text_to_print, f'at GPU{d}')
	except RuntimeError:
		print(text_to_print, 'at CPU')
		pass


class PipelineableSAGEConv(MessagePassing):

	def __init__(self, x, rank, layer, in_channels, out_channels, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.rank = rank
		self.x = x

		self.conv = SAGEConv(in_channels, out_channels, *args, *kwargs)
		self.layer = layer

	def reset_parameters(self):
		self.conv.reset_parameters()

	def test(self, subset, edge_index):
		node_mask = subset
		print("EDGE 0 MASK", node_mask[edge_index[0]])
		print("EDGE 1 MASK", node_mask[edge_index[1]])
		edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
		edge_index = edge_index[:, edge_mask]

	def forward(self, x_edgs):
		if self.training:
			# x, edge_index = x_adjs

			x, n_id_sources, n_id_targets, edge_indexes0, edge_indexes1, size_2 = x_edgs

			print("arg sizes", x.size(), n_id_sources.size(), n_id_targets.size(), edge_indexes0.size(), edge_indexes1.size(), size_2.size())

			device = self.conv.lin_l.weight.get_device()
			if self.layer == 0:
				edge_index = edge_indexes0
				# Chunking requires adds another layer, index arguments first to get rid of it
				if len(list(n_id_sources.size())) > 1:
					n_id_sources = n_id_sources[0]
					n_id_targets = n_id_targets[0]
					print("SOURCE NODES", n_id_sources)
					print("EDGE INDEX", edge_indexes0)

					self.test(n_id_sources, edge_indexes0)
					edge_index, _ = subgraph(n_id_sources, edge_indexes0)

				nid_s = n_id_sources.cpu()
				nid_t = n_id_targets.cpu()
				x_target = self.x[nid_t].to(device)
				x_s = self.x[nid_s].to(device)

				edge_index = edge_index.to(device)
			else:
				edge_index = edge_indexes1
				# Chunking requires adding another layer, index arguments first to get rid of it
				if len(list(n_id_sources.size())) > 1:
					size_2 = size_2[0]
					edge_index, _ = subgraph(n_id_targets, edge_indexes1)
				x_target = x[:size_2]
				x_s = x
				edge_index = edge_index.to(device)

			print('b SAGE', x_s.size(), x_target.size())

			after_SAGE = self.conv((x_s, x_target), edge_index)
			print('a SAGE', after_SAGE.size())


			# nid_t0, nid_t1, nid_s0, nid_s1, edge0, edge1 = x_edgs

			# x, edj0, edj1, size0, size1 = x_edgs

			# Calculate the right layers
			# edge_index, _, size = adjs[self.layer]
			# edge_index = edj0 if self.layer == 0 else edj1
			# size = size0 if self.layer == 0 else size1

			# Experiments using ni_ds and global graph
			# nid_t = nid_t0 if self.layer == 0 else nid_t1
			# nid_s = nid_s0 if self.layer == 0 else nid_s1
			# edge_idx = edge0 if self.layer == 0 else edge1
			#
			# device = self.conv.lin_l.weight.get_device()
			#
			# nid_t = nid_t.squeeze().cpu()
			# nid_s = nid_s.squeeze().cpu()
			# print('Got here')
			#
			# x_target = self.x[nid_t].to(device)
			# x_s = self.x[nid_s].to(device)
			# edge_idx.to(device)

			# x_target = x[:size]

			# print_device(x, 'x')
			# print_device(x_target, 'x_target')
			# print_device(edge_index, 'edge_index')

			# if self.rank == 0:
			# 	print(f'layer:{self.layer}', x.size(), edge_index[1].size(0))
			# print_device(self.conv.lin_l.weight, 'SAGEConv weights')
			# print_device(self.conv.lin_l.bias, 'SAGEConv biias')
			# after_SAGE = self.conv((x, x_target), edge_index)
			# print('x', self.x.size())
			# print('n_s', nid_s.size())
			# print("b_sage", x_s.size(), x_target.size())
			# after_SAGE = self.conv((x_s, x_target), edge_idx)
			# print("a_sage", after_SAGE.size())
			# if self.rank == 0:
			# 	print(f'layer:{self.layer}, after_SAGE:', after_SAGE.size())

			# return after_SAGE, edj0, edj1, size0, size1
			# return after_SAGE, nid_t0, nid_t1, nid_s0, nid_s1, edge0, edge1
			return after_SAGE, n_id_sources, n_id_targets, edge_indexes0, edge_indexes1, size_2

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

	def forward(self, x_edgs):
		# x, adjs = x_adjs
		# x, edj0, edj1, size0, size1 = x_edgs

		# after_SAGE, nid_t0, nid_t1, nid_s0, nid_s1, edge0, edge1 = x_edgs
		x, n_id_sources, n_id_targets, edge_indexes0, edge_indexes1, size_2 = x_edgs

		return F.relu(x, inplace=self.inplace), n_id_sources, n_id_targets, edge_indexes0, edge_indexes1, size_2


class _DropoutNd(Module):
	__constants__ = ['p', 'inplace']
	p: float
	inplace: bool

	def __init__(self, x, layer, p: float = 0.5, inplace: bool = False) -> None:
		super(_DropoutNd, self).__init__()
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, "
											 "but got {}".format(p))
		self.p = p
		self.inplace = inplace
		self.x = x
		self.layer = layer

	def extra_repr(self) -> str:
		return 'p={}, inplace={}'.format(self.p, self.inplace)


class ModifiedDropOut(_DropoutNd):

	def forward(self, x_edgs):
		# x, adjs = x_adjs
		# after_SAGE, nid_t0, nid_t1, nid_s0, nid_s1, edge0, edge1 = x_edgs
		x, n_id_sources, n_id_targets, edge_indexes0, edge_indexes1, size_2 = x_edgs

		# nid_t = nid_t0 if self.layer == 0 else nid_t1
		#
		# after_SAGE = F.dropout(after_SAGE, self.p, self.training, self.inplace)
		# after_SAGE_cpu = after_SAGE.cpu()
		#
		# n_id = nid_t.squeeze().cpu()
		# print("N_ID_T", n_id.size())
		#
		# self.x[n_id] = after_SAGE_cpu

		return F.dropout(x, self.p, self.training, self.inplace), n_id_sources, n_id_targets, edge_indexes0, edge_indexes1, size_2


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

	def forward(self, x_edgs):
		# x, adjs = x_adjs
		# x, edj0, edj1, size0, size1 = x_edgs
		# after_SAGE, nid_t0, nid_t1, nid_s0, nid_s1, edge0, edge1 = x_edgs

		x, n_id_sources, n_id_targets, edge_indexes0, edge_indexes1, size_2 = x_edgs

		return F.log_softmax(x, self.dim, _stacklevel=5)

	def extra_repr(self):
		return 'dim={dim}'.format(dim=self.dim)


def run(rank, world_size,chunk_num, data_split, edge_index, x, y, num_features, num_classes):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	# dist.init_process_group('nccl', rank=rank, world_size=world_size)
	# torch.torch.cuda.set_device(rank)
	# init_rpc(f'worker{rank}', rank=rank, world_size=world_size)

	train_mask, val_mask, test_mask = data_split
	train_idx = train_mask.nonzero(as_tuple=False).view(-1)
	train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

	train_loader = NeighborSampler(edge_index, node_idx=train_idx,
																 sizes=[25, 10], batch_size=1024,
																 shuffle=True, persistent_workers=True,
																 num_workers=24 // world_size)

	if rank == 0:
		subgraph_loader = NeighborSampler(edge_index, node_idx=None,
																			sizes=[-1], batch_size=2048,
																			shuffle=False, num_workers=6)

	torch.manual_seed(12345)
	# model = SAGE(num_features, 256, num_classes).to(rank)
	hidden_channels = 256
	model = Sequential(
		PipelineableSAGEConv(x=x, rank=rank, layer=0, in_channels=num_features, out_channels=hidden_channels),
		ModifiedReLU(),
		ModifiedDropOut(x=x, p=0.5, layer=0),
		PipelineableSAGEConv(x=x, rank=rank, layer=1, in_channels=hidden_channels, out_channels=num_classes),
		ModifiedLogMax(dim=-1)
	)
	# model.to(rank)

	if rank == 0:
		print(model)

	# model = GPipe(model, balance=[1, 2, 2], chunks=1, checkpoint='never')

	# TODO change input to tensor type (concatenate each batch with indexes, and sizes)?

	# model = Pipe(model, chunks=1, checkpoint='never')
	model = GPipe(model, balance=[1, 2, 2], chunks=chunk_num, checkpoint='never')
	# model = DistributedDataParallel(model, device_ids=[rank])

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	# Simulate cases those data can not be fully stored by GPU memory
	x, y = x, y.to(rank)

	for epoch in range(1, 21):
		model.train()
		epoch_start = time.time()
		for batch_size, n_id, adjs in train_loader:

			# if rank == 0:
			# for adj in adjs:
			# 	print(adj.size[1])

			sizes = [adj.size[1] for adj in adjs]
			sizes = [torch.tensor([size]).to(rank) for size in sizes]
			# sizes = [torch.tensor([size]) for size in sizes]
			# print(sizes)

			adjs = [adj.edge_index for adj in adjs]
			adjs = [adj.to(rank) for adj in adjs]

			# adjs = torch.stack(adjs).to(rank)

			optimizer.zero_grad()

			# TODO calculate x and x_target manually for all sampling layer, make edge_index i.e. adjs global (or not)!
			# TODO using chunk_num, multiply edge_index across manually
			# TODO using chunk_num
			# print(n_id.size())

			# n_id_targets_list = []
			# n_id_sources_list = []

			# for size in sizes:
			# 	n_id_targets = n_id[:size]
			# 	n_id_sources_only = n_id[size:]
			#
			# 	n_id_targets = torch.chunk(n_id_targets, chunks=chunk_num)
			# 	n_id_targets = torch.stack(n_id_targets)
			#
			# 	n_id_sources_only = torch.chunk(n_id_sources_only, chunks=chunk_num)
			# 	n_id_sources_only = torch.stack(n_id_sources_only)
			#
			# 	# print("1", n_id_sources_only.size(), n_id_targets.size())
			#
			# 	if chunk_num != 1:
			# 		n_id_targets = n_id_targets.squeeze()
			# 		n_id_sources_only = n_id_sources_only.squeeze()
			#
			# 	# print("2", n_id_sources_only.size(), n_id_targets.size())
			# 	n_id_sources = torch.cat((n_id_targets, n_id_sources_only), dim=1)
			#
			# 	n_id_targets_list.append(n_id_targets)
			# 	n_id_sources_list.append(n_id_sources)

			# 1st layer information
			n_id_targets = n_id[:sizes[0]]
			n_id_sources_only = n_id[sizes[0]:]
			n_id_sources = n_id

			# 2nd layer information
			# We just want to divide the size by chunk_num
			size_2 = torch.div(sizes[1], chunk_num, rounding_mode='trunc')

			# Chunk manually for chunk_num > 1
			if chunk_num > 1:
				n_id_targets = torch.chunk(n_id_targets, chunks=chunk_num)

				# Pad last chunk with a random n_id if chunking is not even: pad n_id_targets[0] - n_id_targets[1] so it is even
				n_id_targs = n_id[:sizes[0]]
				padding_choice = n_id_targs[torch.randint(len(n_id_targs), (1,))]
				n_id_targets = list(n_id_targets)
				n_id_targets[-1] = F.pad(n_id_targets[-1], (0, len(n_id_targets[0]) - len(n_id_targets[1])), "constant",
																int(padding_choice))

				n_id_targets = torch.stack(n_id_targets)
				n_id_sources_only = torch.chunk(n_id_sources_only, chunks=chunk_num)

				# Pad last chunk with a random n_id if chunking is not even: pad n_id_sources_only[0] - n_id_sources_only[1] so it is even
				n_id_srcs = n_id[sizes[0]:]
				padding_choice = n_id_srcs[torch.randint(len(n_id_srcs), (1,))]
				n_id_sources_only = list(n_id_sources_only)
				n_id_sources_only[-1] = F.pad(n_id_sources_only[-1], (0, len(n_id_sources_only[0]) - len(n_id_sources_only[1])), "constant",
																int(padding_choice))

				n_id_sources_only = torch.stack(n_id_sources_only)
				# print("1", n_id_sources_only.size(), n_id_targets.size())

				n_id_targets = n_id_targets.squeeze()
				n_id_sources_only = n_id_sources_only.squeeze()

				# print("2", n_id_sources_only.size(), n_id_targets.size())

				n_id_sources = torch.cat((n_id_targets, n_id_sources_only), dim=1)

				# Let every micro-batch know what size_2 is
				size_2 = size_2.repeat(chunk_num, 1)

			edge_indexes = []
			for adj in adjs:
				edge_index = adj.repeat(chunk_num, 1)
				edge_indexes.append(edge_index)

			print("model args", torch.empty(0).size(), n_id_sources.size(), n_id_targets.size(),
						edge_indexes[0].size(), edge_indexes[1].size(), size_2.size())

			out = model((
				torch.empty(0),
				n_id_sources.to(rank),
				n_id_targets.to(rank),
				edge_indexes[0],
				edge_indexes[1],
				size_2
			))

			# out = model(
			# 	(n_id_targets_list[0].to(rank),
			# 	 n_id_targets_list[1].to(rank),
			# 	 n_id_sources_list[0].to(rank),
			# 	 n_id_sources_list[1].to(rank),
			# 	 edge_indexes[0],
			# 	 edge_indexes[1])
			# )

			# out = model((x[n_id].to(rank), adjs[0], adjs[1], sizes[0], sizes[1]))
			# out = model((x[n_id], adjs[0], adjs[1], sizes[0], sizes[1]))
			#
			# print("YSIZE",y.size())
			# print("N_ID SIZE",n_id.size())
			# print("Batch_size", batch_size)
			# print("YSIZE",y.size())
			# print("N_id",n_id[:batch_size].size())
			#
			# print("MAX", torch.max(n_id[:batch_size]))

			# node_ids = n_id
			# filter_max = node_ids < y.size(0)
			# node_ids = node_ids[torch.nonzero(filter_max)][:batch_size]
			# print("Max 2", torch.max(node_ids))

			# print("OUT", out.size(), y[n_id[:batch_size]].size())
			# loss = F.nll_loss(out, y[node_ids])
			# print_device(out, "out")
			# print_device(y, "y")
			# print_device(n_id, "n_id")

			out = out.to(rank)
			# print("OUT", out.size())
			# print("LABEL", y[n_id[:batch_size]].size())

			loss = F.nll_loss(out, y[n_id[:batch_size]])

			loss.backward()
			optimizer.step()

		# dist.barrier()

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

	# if epoch % 5 == 0:
	# 	model.eval()
	# 	with torch.no_grad():
	# 		out = evaluate(model.module, x, rank, subgraph_loader)
	# 	res = out.argmax(dim=-1) == y
	# 	acc1 = int(res[train_mask].sum()) / int(train_mask.sum())
	# 	acc2 = int(res[val_mask].sum()) / int(val_mask.sum())
	# 	acc3 = int(res[test_mask].sum()) / int(test_mask.sum())
	# 	print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

	# dist.barrier()


# dist.destroy_process_group()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_chunks', type=int, default=1)
	args = parser.parse_args()

	dataset = Reddit('/data/terencehernandez/Reddit')
	data = dataset[0]
	world_size = 3  # torch.cuda.device_count()
	print('Let\'s use', world_size, 'GPUs!')
	data_split = (data.train_mask, data.val_mask, data.test_mask)
	# mp.spawn(run,
	# 				 args=(world_size, data_split, data.edge_index, data.x, data.y, dataset.num_features, dataset.num_classes),
	# 				 nprocs=world_size, join=True)

	run(0, world_size, args.num_chunks, data_split, data.edge_index, data.x, data.y, dataset.num_features, dataset.num_classes)
