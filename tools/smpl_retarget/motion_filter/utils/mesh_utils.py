import trimesh
import os.path as osp
import numpy as np
import torch
import os
import sys
import torch.nn as nn

ESSENTIALS_DIR = 'data/essentials'


def sparse_batch_mm(m1, m2):
    """

    m1: sparse matrix of size N x M
    m2: dense matrix of size B x M x K
    returns m1@m2 matrix of size B x N x K
    """

    batch_size = m2.shape[0]
    # stack m2 into columns: (B x N x K) -> (N, B, K) -> (N, B * K)
    m2_stack = m2.transpose(0, 1).reshape(m1.shape[1], -1)
    result = m1.mm(m2_stack).reshape(m1.shape[0], batch_size, -1) \
        .transpose(1, 0)
    return result


class HDfier():
    def __init__(self, model_type='smpl', device='cuda'):
        hd_operator_path = osp.join(ESSENTIALS_DIR, 'hd_model', model_type,
                                    f'{model_type}_neutral_hd_vert_regressor_sparse.npz')
        hd_operator = np.load(hd_operator_path)
        self.hd_operator = torch.sparse.FloatTensor(
            torch.tensor(hd_operator['index_row_col']),
            torch.tensor(hd_operator['values']),
            torch.Size(hd_operator['size']))
        self.model_type = model_type
        self.device = device

    def hdfy_mesh(self, vertices):
        """
        Applies a regressor that maps SMPL vertices to uniformly distributed vertices
        """
        # device = body.vertices.device
        # check if vertices ndim are 3, if not , add a new axis
        if vertices.dim() != 3:
            # batchify the vertices
            vertices = vertices[None, :, :]

        # check if vertices are an ndarry, if yes, make pytorch tensor
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(self.device)

        vertices = vertices.to(torch.double)

        if self.hd_operator.device != vertices.device:
            self.hd_operator = self.hd_operator.to(vertices.device)
        hd_verts = sparse_batch_mm(self.hd_operator, vertices).to(torch.float)
        return hd_verts
