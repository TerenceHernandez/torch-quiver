import torch

CHUNK_NUM = 32


def partition_without_replication(device, probs, ids):
    """Partition node with given node IDs and node access distribution.
    The result will cause no replication between each parititon.
    We assume node IDs can be placed in the given device.
    Args:
        device (int): device which computes the partitioning strategy
        probs (torch.Tensor): node access distribution
        ids (Optional[torch.Tensor]): specified node IDs
    Returns:
        [torch.Tensor]: list of IDs for each partition
    """

		# probs dim = (host, num_node_per_gpu), ids = max_num_node_per_host
    ranks = len(probs)
    if ids is not None:
        ids = ids.to(device)

		# Send all probs to device
    probs = [
        prob[ids].to(device) if ids is not None else prob.to(device)
        for prob in probs
    ]

		# max_num_node_per_host
    total_size = ids.size(0) if ids is not None else probs[0].size(0)
    res = [None] * ranks
    for rank in range(ranks):
        res[rank] = []
    CHUNK_SIZE = (total_size + CHUNK_NUM - 1) // CHUNK_NUM
    chunk_beg = 0
    beg_rank = 0
    for i in range(CHUNK_NUM):
        chunk_end = min(total_size, chunk_beg + CHUNK_SIZE)
        chunk_size = chunk_end - chunk_beg
        chunk = torch.arange(chunk_beg,
                             chunk_end,
                             dtype=torch.int64,
                             device=device)
        probs_sum_chunk = [
            torch.zeros(chunk_size, device=device) + 1e-6 for i in range(ranks)
        ]
        for rank in range(ranks):
            for dst_rank in range(ranks):
                # Add own probabilities multiplied by num_hosts
                if dst_rank == rank:
                    probs_sum_chunk[rank] += probs[dst_rank][chunk] * ranks
                else:
                # Take away other host's probabilities
                    probs_sum_chunk[rank] -= probs[dst_rank][chunk]

        # acc_size is accumulator representing how much has been allocated of the chunk
        acc_size = 0
        # rank_size = split a chunk into different ranks (portion of chunk that will be split into rank)
        rank_size = (chunk_size + ranks - 1) // ranks
        picked_chunk_parts = torch.LongTensor([]).to(device)
        for rank_ in range(beg_rank, beg_rank + ranks):
            rank = rank_ % ranks
            probs_sum_chunk[rank][picked_chunk_parts] -= 1e6
            rank_size = min(rank_size, chunk_size - acc_size) # makes sure we don't go over (but allows trailing bits to be added)
            _, rank_order = torch.sort(probs_sum_chunk[rank], descending=True)
            pick_chunk_part = rank_order[:rank_size]
            pick_ids = chunk[pick_chunk_part]
            picked_chunk_parts = torch.cat(
                (picked_chunk_parts, pick_chunk_part))  # append chunk portion to growing list
            res[rank].append(pick_ids)  # track of which part of the chunk is added to which rank
            acc_size += rank_size
        beg_rank += 1
        chunk_beg += chunk_size
    for rank in range(ranks):
        # res is the allocation of chunk portions to ranks/hosts
        res[rank] = torch.cat(res[rank])
        if ids is not None:
            res[rank] = ids[res[rank]]
            # res is the allocation of node_ids to ranks/hosts (rank, node_ids)

    return res


def partition_with_replication(device, probs, ids, per_rank_size):
    """Partition node with given node IDs and node access distribution.
    The result will cause replication between each parititon,
    but the size of each partition will not exceed per_rank_size.
    """
    partition_res = partition_without_replication(device, probs, ids)
    if ids is not None:
        ids = ids.to(device)
    ranks = len(probs)
    total_res = [
        torch.empty(per_rank_size, device=device) for i in range(ranks)
    ]
    probs = [prob.clone().to(device) for prob in probs]
    for rank in range(ranks):
        partition_ids = partition_res[rank]
        probs[rank][partition_ids] = -1e6
        replication_size = per_rank_size - partition_ids.size(0)
        _, prev_order = torch.sort(probs[rank], descending=True)
        replication_ids = ids[
            prev_order[:
                       replication_size]] if ids is not None else prev_order[:
                                                                             replication_size]
        total_res[rank] = torch.cat((partition_ids, replication_ids))
    return total_res


def select_nodes(device, probs, ids):
    nodes = probs[0].size(0)
    prob_sum = torch.zeros(nodes, device=device)
    for prob in probs:
        if ids is None:
            prob_sum += prob
        else:
            prob_sum[ids] += prob[ids]
    node_ids = torch.nonzero(prob_sum)
    return prob_sum, node_ids


def partition_free(device, probs, ids, per_rank_size):
    """Partition node with given node IDs and node access distribution.
    The result will cause either replication or missing nodes across partitions.
    The size of each partition is limited by per_rank_size.
    """
    prob_sum, node_ids = select_nodes(device, probs, ids)
    nodes = node_ids.size(0)
    ranks = len(probs)
    limit = ranks * per_rank_size
    if nodes <= limit:
        return partition_with_replication(device, probs, node_ids,
                                          per_rank_size), None
    else:
        _, prev_order = torch.sort(prob_sum, descending=True)
        limit_ids = prev_order[:limit]
        return partition_without_replication(device, probs,
                                             node_ids),