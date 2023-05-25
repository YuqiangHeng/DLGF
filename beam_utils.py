"""
@author: Yuqiang (Ethan) Heng
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from typing import (
    Tuple, Optional, Union, Any, Sequence, TYPE_CHECKING
)
from torch.testing._internal.common_dtype import integral_types
from torch import Tensor

"""
implementation of unravel_index in pytorch, its behavior is identical to numpy.unravel_index()
https://github.com/krshrimali/pytorch/blob/impl/unravel_index/torch/functional.py
"""
def unravel_index(
    indices: Tensor,
    shape: Union[int, Sequence, Tensor],
    *,
    as_tuple: bool = True
) -> Union[Tuple[Tensor, ...], Tensor]:
    r"""Converts a `Tensor` of flat indices into a `Tensor` of coordinates for the given target shape.
    Args:
        indices: An integral `Tensor` containing flattened indices of a `Tensor` of dimension `shape`.
        shape: The shape (can be an `int`, a `Sequence` or a `Tensor`) of the `Tensor` for which
               the flattened `indices` are unraveled.
    Keyword Args:
        as_tuple: A boolean value, which if `True` will return the result as tuple of Tensors,
                  else a `Tensor` will be returned. Default: `True`
    Returns:
        unraveled coordinates from the given `indices` and `shape`. See description of `as_tuple` for
        returning a `tuple`.
    .. note:: The default behaviour of this function is analogous to
              `numpy.unravel_index <https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html>`_.
    Example::
        >>> indices = torch.tensor([22, 41, 37])
        >>> shape = (7, 6)
        >>> torch.unravel_index(indices, shape)
        (tensor([3, 6, 6]), tensor([4, 5, 1]))
        >>> torch.unravel_index(indices, shape, as_tuple=False)
        tensor([[3, 4],
                [6, 5],
                [6, 1]])
        >>> indices = torch.tensor([3, 10, 12])
        >>> shape_ = (4, 2, 3)
        >>> torch.unravel_index(indices, shape_)
        (tensor([0, 1, 2]), tensor([1, 1, 0]), tensor([0, 1, 0]))
        >>> torch.unravel_index(indices, shape_, as_tuple=False)
        tensor([[0, 1, 0],
                [1, 1, 1],
                [2, 0, 0]])
    """
    def _helper_type_check(inp: Union[int, Sequence, Tensor], name: str):
        # `indices` is expected to be a tensor, while `shape` can be a sequence/int/tensor
        if name == "shape" and isinstance(inp, Sequence):
            for dim in inp:
                if not isinstance(dim, int):
                    raise TypeError("Expected shape to have only integral elements.")
                if dim < 0:
                    raise ValueError("Negative values in shape are not allowed.")
        elif name == "shape" and isinstance(inp, int):
            if inp < 0:
                raise ValueError("Negative values in shape are not allowed.")
        elif isinstance(inp, Tensor):
            if inp.dtype not in integral_types():
                raise TypeError(f"Expected {name} to be an integral tensor, but found dtype: {inp.dtype}")
            if torch.any(inp < 0):
                raise ValueError(f"Negative values in {name} are not allowed.")
        else:
            allowed_types = "Sequence/Scalar (int)/Tensor" if name == "shape" else "Tensor"
            msg = f"{name} should either be a {allowed_types}, but found {type(inp)}"
            raise TypeError(msg)

    _helper_type_check(indices, "indices")
    _helper_type_check(shape, "shape")

    # Convert to a tensor, with the same properties as that of indices
    if isinstance(shape, Sequence):
        shape_tensor: Tensor = indices.new_tensor(shape)
    elif isinstance(shape, int) or (isinstance(shape, Tensor) and shape.ndim == 0):
        shape_tensor = indices.new_tensor((shape,))
    else:
        shape_tensor = shape

    # By this time, shape tensor will have dim = 1 if it was passed as scalar (see if-elif above)
    assert shape_tensor.ndim == 1, "Expected dimension of shape tensor to be <= 1, "
    f"but got the tensor with dim: {shape_tensor.ndim}."

    # In case no indices passed, return an empty tensor with number of elements = shape.numel()
    if indices.numel() == 0:
        # If both indices.numel() == 0 and shape.numel() == 0, short-circuit to return itself
        shape_numel = shape_tensor.numel()
        if shape_numel == 0:
            raise ValueError("Got indices and shape as empty tensors, expected non-empty tensors.")
        else:
            output = [indices.new_tensor([]) for _ in range(shape_numel)]
            return tuple(output) if as_tuple else torch.stack(output, dim=1)

    if torch.max(indices) >= torch.prod(shape_tensor):
        raise ValueError("Target shape should cover all source indices.")

    coefs = shape_tensor[1:].flipud().cumprod(dim=0).flipud()
    coefs = torch.cat((coefs, coefs.new_tensor((1,))), dim=0)
    coords = torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape_tensor

    if as_tuple:
        return tuple(coords[..., i] for i in range(coords.size(-1)))
    return coords

def pow_2_dB(x):
    return 10*np.log10(x)
def dB_2_pow(x):
    return 10**(x/10)

def DFT_angles(n_beam):
    theta_max = 1.0
    delta_theta = theta_max*2/n_beam
    if n_beam % 2 == 1:
        thetas = np.arange(0,theta_max,delta_theta)
        thetas = np.concatenate((-np.flip(thetas[1:]),thetas))
    else:
        thetas = np.arange(delta_theta/2,theta_max,delta_theta) 
        thetas = np.concatenate((-np.flip(thetas),thetas))
    return thetas

def ULA_DFT_codebook(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = DFT_angles(nseg)
    for i,theta in enumerate(thetas):
        arr_response_vec = [-1j*2*np.pi*k*spacing*theta for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)   
    return codebook_all

def UPA_DFT_codebook(azimuth_min=0,azimuth_max=1,elevation_min=0,elevation_max=1,n_azimuth=16,n_elevation=16,n_antenna_azimuth=8,n_antenna_elevation=8,spacing=0.5):
    codebook_all = np.zeros((n_azimuth,n_elevation,n_antenna_azimuth*n_antenna_elevation),dtype=np.complex_)

    azimuths = np.linspace(azimuth_min,azimuth_max,n_azimuth,endpoint=False)
    elevations = np.linspace(elevation_min,elevation_max,n_elevation,endpoint=False)

    a_azimuth = np.tile(azimuths*elevations,(n_antenna_azimuth,1)).T
    a_azimuth = 1j*2*np.pi*a_azimuth
    a_azimuth = a_azimuth * np.tile(np.arange(n_antenna_azimuth),(n_azimuth,1))
    a_azimuth = np.exp(a_azimuth)/np.sqrt(n_antenna_azimuth)  

    a_elevation = np.tile(elevations,(n_antenna_elevation,1)).T
    a_elevation = 1j*2*np.pi*a_elevation
    a_elevation = a_elevation * np.tile(np.arange(n_antenna_elevation),(n_elevation,1))
    a_elevation = np.exp(a_elevation)/np.sqrt(n_antenna_elevation)  
    
    codebook_all = np.kron(a_azimuth,a_elevation)
    return codebook_all

def calc_beam_pattern(beam, phi_min=-np.pi/2, phi_max=np.pi/2, resolution = int(1e3), n_antenna = 64, array_type='ULA', k=0.5):
    phi_all = np.linspace(phi_min,phi_max,resolution)
    array_response_vectors = np.tile(phi_all,(n_antenna,1)).T
    array_response_vectors = -1j*2*np.pi*k*np.sin(array_response_vectors)
    array_response_vectors = array_response_vectors * np.arange(n_antenna)
    array_response_vectors = np.exp(array_response_vectors)/np.sqrt(n_antenna)
    gains = abs(array_response_vectors.conj() @ beam)**2
    return phi_all, gains
    
def uniform_codebook(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = np.linspace(-np.pi/2,np.pi/2,nseg)    
    codebook_all = np.tile(thetas,(n_antenna,1)).T
    codebook_all = -1j*2*np.pi*spacing*np.sin(codebook_all)
    codebook_all = codebook_all * np.arange(n_antenna)
    codebook_all = np.exp(codebook_all)/np.sqrt(n_antenna)
    return thetas,codebook_all

def array_response_vector(theta,n_antenna,spacing=0.5):
    beam = -1j*2*np.pi*spacing*np.sin(theta)*np.arange(n_antenna)
    beam = np.exp(beam)/np.sqrt(n_antenna)
    return beam
    
def get_AMCF_beam(omega_min, omega_max, n_antenna=64,Q=2**10): 
    #omega_min and omega_max are the min and max beam coverage, Q is the resolution
    q = np.arange(Q)+1
    omega_q = -1 + (2*q-1)/Q
    #array response matrix
    A_phase = np.outer(np.arange(n_antenna),omega_q)
    A = np.exp(1j*np.pi*A_phase)
    #physical angles between +- 90 degree
    theta_min, theta_max = np.arcsin(omega_min),np.arcsin(omega_max)
    #beamwidth in spatial angle
    B = omega_max - omega_min
    mainlobe_idc = ((omega_q >= omega_min) & (omega_q <= omega_max)).nonzero()[0]
    sidelobe_idc = ((omega_q < omega_min) | (omega_q > omega_max)).nonzero()[0]
    #ideal beam amplitude pattern
    g = np.zeros(Q)
    g[mainlobe_idc] = np.sqrt(2/B)
    #g_eps = g in mainlobe and = eps in sidelobe to avoid log0=Nan
    eps = 2**(-52)
    g_eps = g
    g_eps[sidelobe_idc] = eps
    
    v0_phase = B*np.arange(n_antenna)*np.arange(1,n_antenna+1)/2/n_antenna + np.arange(n_antenna)*omega_min
    v0 = 1/np.sqrt(n_antenna)*np.exp(1j*np.pi*v0_phase.conj().T)
    v = v0
    ite = 1
    mse_history = []
    while True:
        mse = np.power(abs(A.conj().T @ v) - g,2).mean()
        mse_history.append(mse)
        if ite >= 10 and abs(mse_history[-1] - mse_history[-2]) < 0.01*mse_history[-1]:
            break
        else:
            ite += 1
        Theta = np.angle(A.conj().T @ v)
        r = g * np.exp(1j*Theta)
        v = 1/np.sqrt(n_antenna)*np.exp(1j*np.angle(A @ r))
    return v
        
def AMCF_boundaries(n_beams):
    beam_boundaries = np.zeros((n_beams,2))
    for k in range(n_beams):
        beam_boundaries[k,0] = -1 + k*2/n_beams
        beam_boundaries[k,1] = beam_boundaries[k,0] + 2/n_beams
    return beam_boundaries

def get_AMCF_codebook(n_beams,n_antenna):
    AMCF_codebook_all = np.zeros((n_beams,n_antenna),dtype=np.complex_)
    AMCF_boundary = AMCF_boundaries(n_beams)
    for i in range(n_beams):
        AMCF_codebook_all[i,:] = get_AMCF_beam(AMCF_boundary[i,0], AMCF_boundary[i,1],n_antenna=n_antenna)
    return AMCF_codebook_all

class Node():
    def __init__(self, n_antenna:int, n_beam:int, codebook:np.ndarray, beam_index:np.ndarray, noise_power=0):
        super(Node, self).__init__()
        self.codebook = codebook
        self.n_antenna = n_antenna
        self.n_beam = n_beam
        self.beam_index = beam_index # indices of the beams (in the same layer) in the codebook
        self.noise_power = noise_power
        self.parent = None
        self.child = None
        
    def forward(self, h):
        bf_signal = np.matmul(h, self.codebook.conj().T)
        noise_real = np.random.normal(loc=0,scale=1,size=bf_signal.shape)*np.sqrt(self.noise_power/2)
        noise_imag = np.random.normal(loc=0,scale=1,size=bf_signal.shape)*np.sqrt(self.noise_power/2)
        bf_signal = bf_signal + noise_real + 1j*noise_imag
        bf_gain = np.power(np.absolute(bf_signal),2)
        # bf_gain = np.power(np.absolute(np.matmul(h, self.codebook.conj().T)),2)
        return bf_gain
    
    def get_beam_index(self):
        return self.beam_index

    def set_child(self, child):
        self.child = child
        
    def set_parent(self, parent):
        self.parent = parent
        
    def get_child(self):
        return self.child
    
    def get_parent(self):
        return self.parent
    
    def is_leaf(self):
        return self.get_child() is None
    
    def is_root(self):
        return self.get_parent() is None
    
class Beam_Search_Tree():
    def __init__(self, n_antenna, n_narrow_beam, k, noise_power):
        super(Beam_Search_Tree, self).__init__()
        assert math.log(n_narrow_beam,k).is_integer()
        self.n_antenna = n_antenna
        self.k = k #number of beams per branch per layer
        self.n_layer = int(math.log(n_narrow_beam,k))
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = noise_power
        self.beam_search_candidates = []
        for l in range(self.n_layer):
            self.beam_search_candidates.append([])
        self.nodes = []
        for l in range(self.n_layer):
            n_nodes = k**l
            n_beams = k**(l+1)
            if l == self.n_layer-1:
                beams = ULA_DFT_codebook(nseg=n_beams,n_antenna=n_antenna)
            else:                    
                beam_boundaries = AMCF_boundaries(n_beams)
                beams = np.array([get_AMCF_beam(omega_min=beam_boundaries[i,0], omega_max=beam_boundaries[i,1], n_antenna = n_antenna) for i in range(n_beams)])
                beams = np.flipud(beams)
            beam_idx_per_codebook = [np.arange(i,i+k) for i in np.arange(0,n_beams,k)]
            codebooks = [beams[beam_idx_per_codebook[i]] for i in range(n_nodes)]
            nodes_cur_layer = []
            nodes_cur_layer = [Node(n_antenna=n_antenna,n_beam = k, codebook=codebooks[i], beam_index=beam_idx_per_codebook[i], noise_power=self.noise_power) for i in range(n_nodes)]
            self.nodes.append(nodes_cur_layer)
            if l > 0:
                parent_nodes = self.nodes[l-1]
                for p_i, p_n in enumerate(parent_nodes):
                    child_nodes = nodes_cur_layer[p_i*k:(p_i+1)*k]
                    p_n.set_child(child_nodes)
                    for c_n in child_nodes:
                        c_n.set_parent(p_n)
        self.root = self.nodes[0][0]
        
    def forward(self, h):
        cur_node = self.root
        while not cur_node.is_leaf():
            bf_gain = cur_node.forward(h)
            next_node_idx = bf_gain.argmax()
            cur_node = cur_node.get_child()[next_node_idx]
        nb_bf_gain = cur_node.forward(h)
        max_nb_bf_gain = nb_bf_gain.max()
        max_nb_idx_local = nb_bf_gain.argmax()
        max_nb_idx_global = cur_node.get_beam_index()[max_nb_idx_local]
        return max_nb_bf_gain, max_nb_idx_global        
        
    def forward_batch(self, hbatch):
        bsize, in_dim = hbatch.shape
        max_nb_idx_batch = np.zeros(bsize,dtype=np.int32)
        max_nb_bf_gain_batch = np.zeros(bsize)
        for b_idx in range(bsize):
            h = hbatch[b_idx]
            nb_gain,nb_idx = self.forward(h)
            max_nb_idx_batch[b_idx] = nb_idx
            max_nb_bf_gain_batch[b_idx] = nb_gain
        return max_nb_bf_gain_batch, max_nb_idx_batch
    