import torch


def is_traceless(x: torch.Tensor) -> bool:
    if len(x.shape[1:-1]) < 2:
        return True
    elif len(x.shape[1:-1]) == 2:
        _is_traceless = (torch.Tensor([abs(x[i,:, :, j].trace()) for i in range(x.shape[0]) for j in range(x.shape[-1])]) < 1e-14).all()
        return _is_traceless
    elif len(x.shape[1:-1]) == 3:
        _is_traceless = (torch.Tensor([x[i, j, :, :, k].trace() for i in range(x.shape[0]) for j in range(x.shape[1]) for k in range(x.shape[-1])]) < 1e-14).all()
        return _is_traceless
    else:
        raise NotImplementedError
    

def is_symmetric(x: torch.Tensor) -> bool:
    if len(x.shape[1:-1]) < 2:
        return True
    elif len(x.shape[1:-1]) == 2:
        _is_symmetric = torch.stack([abs(x[i, :, :, j] - x[i, :, :, j].permute(1, 0)).sum() < 1e-14 for i in range(x.shape[0]) for j in range(x.shape[-1])]).all()
        return _is_symmetric
    elif len(x.shape[1:-1]) == 3:
        _is_symmetric = torch.stack([abs(x[i, :, :, :, j] - x[i, :, :, :, j].permute(1, 2, 0)).sum() < 1e-14 for i in range(x.shape[0]) for j in range(x.shape[-1])]).all()
        _is_symmetric = _is_symmetric and torch.stack([abs(x[i, :, :, :, j] - x[i, :, :, :, j].permute(2, 0, 1)).sum() < 1e-14 for i in range(x.shape[0]) for j in range(x.shape[-1])]).all()
        return _is_symmetric
    else:
        raise NotImplementedError


def is_normalized(vec: torch.Tensor, x: torch.Tensor):
    assert len(vec.shape) == 3
    if x.shape[1:-1] == (1, ):
        return (abs(x - 1.0) < 1e-14).all()
    elif x.shape[1:-1] == (3, ):
        return (abs(torch.einsum('aiu, aiv -> auv', x, vec) - 1.0) < 1e-14).all()
    elif x.shape[1:-1] == (3, 3, ):
        return (abs(torch.einsum('aiju, aiv, ajv -> auv', x, vec, vec) - 1.0) < 1e-14).all()
    elif x.shape[1:-1] == (3, 3, 3, ):
        return (abs(torch.einsum('aijku, aiv, ajv, akv -> auv', x, vec, vec, vec) - 1.0) < 1e-14).all()
    else:
        raise NotImplementedError
