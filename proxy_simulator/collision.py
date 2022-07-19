import torch

EDGE_TRANSFORM = torch.tensor(
    [
        [-1, 1, 0, 0],
        [0, -1, 1, 0],
        [0, 0, -1, 1],
        [1, 0, 0, -1],
    ],
    dtype=torch.float32,
    device='cuda',
).reshape(1, 4, 4)


AXIS_FLIP = torch.tensor(
    [
        [0, 1],
        [-1, 0],
    ],
    dtype=torch.float32,
    device='cuda',
).reshape(1, 2, 2)


def get_faces(polygon):
    # assumes polygon is a bounding box
    if not EDGE_TRANSFORM.device == polygon.device:
        etf = EDGE_TRANSFORM.to(polygon.device)
    else:
        etf = EDGE_TRANSFORM

    return etf @ polygon


def get_normals(faces):
    if not AXIS_FLIP.device == faces.device:
        af = AXIS_FLIP.to(faces.device)
    else:
        af = AXIS_FLIP
    
    normals = af @ faces.unsqueeze(-1)
    normals = normals.squeeze(-1)

    return normals / torch.linalg.norm(normals, dim=-1, keepdim=True)


def check_collision(polygon_a, polygon_b):
    """
    Checks for collision between pairs of 2D-polygons. Supports
    parallel checks of batches of pairs.
    """
    normals = get_normals(
        torch.cat(
            [get_faces(polygon_a), get_faces(polygon_b)], 
            dim=1
        )
    )

    # project polygons onto normals and check for separation
    polygon_a_proj = polygon_a.view(-1, 1, 4, 2) @ normals.view(-1, 8, 2, 1)
    polygon_b_proj = polygon_b.view(-1, 1, 4, 2) @ normals.view(-1, 8, 2, 1)
    
    polygon_a_proj_min = torch.min(polygon_a_proj, dim=-2)[0]
    polygon_a_proj_max = torch.max(polygon_a_proj, dim=-2)[0]
    polygon_b_proj_min = torch.min(polygon_b_proj, dim=-2)[0]
    polygon_b_proj_max = torch.max(polygon_b_proj, dim=-2)[0]

    not_separable = torch.gt(polygon_a_proj_max, polygon_b_proj_min) & \
            torch.gt(polygon_b_proj_max, polygon_a_proj_min)

    return torch.all(not_separable, dim=-2).view(-1)