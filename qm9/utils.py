import torch


def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == 'qm9':
        return compute_mean_mad_from_dataloader(dataloaders['train'], properties)
    # 手误了 ??？  或许是qm9_second_half和qm9_first_half ???
    elif dataset_name == 'qm9_second_half' or dataset_name == 'qm9_second_half':
        return compute_mean_mad_from_dataloader(dataloaders['valid'], properties)
    else:
        raise Exception('Wrong dataset name')

# 样本与样本均值差的绝对值的平均值，叫做平均绝对偏差（Mean Absolute Deviation，简称MAD）。
def compute_mean_mad_from_dataloader(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
    return property_norms

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)


    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms):
    batch_size, n_nodes, _ = minibatch['positions'].size()
    node_mask = minibatch['atom_mask'].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch[key]
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1:
            # Global feature.
            assert properties.size() == (batch_size,) # 这里是元组的固定写法，等价于tuple([8])，为什么不用tuple(8)或者(8)，因为前者会报错，后者就是数字8
            reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # Node feature.
            assert properties.size()[:2] == (batch_size, n_nodes)

            context_key = properties

            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)

            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=2)
    # Mask disabled nodes!
    context = context * node_mask
    # 总结一下，就是把条件和每一个样本中的每一个原子对应上了，有几个条件就有几维，例如[8, 22, 1]就是有8个分子，每个分子最多22个原子，每个原子都考虑那1个条件
    assert context.size(2) == context_node_nf
    return context
