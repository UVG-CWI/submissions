import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

def eval_pointcloud(pre_mesh_ply, gt_mesh_ply, samplepoint, eval_type, 
                    thresholds=[5, 10, 20]):
        if eval_type == 'real_obj':
            pn_gt = np.loadtxt(gt_mesh_ply, delimiter=' ')
            samplepoint = len(pn_gt)
            pointcloud_tgt = pn_gt[:,:3]
            normals_tgt = pn_gt[:,3:]
            pointcloud, normals = sampleGT(pre_mesh_ply, samplepointsnum = samplepoint)
        elif eval_type == 'ply':
            pointcloud_tgt, normals_tgt = load_ply_points(gt_mesh_ply)
            pointcloud, normals = load_ply_points(pre_mesh_ply)
        else:
            pointcloud_tgt, normals_tgt = sampleGT(gt_mesh_ply, samplepointsnum = samplepoint)
            pointcloud, normals = sampleGT(pre_mesh_ply, samplepointsnum = samplepoint)

        if pointcloud.shape[0] == 0 or pointcloud_tgt.shape[0] == 0:
            out_dict = {
                'N_Acc' : 0,
                'N_Comp' : 0,
                'normals': 0,
                'CD_Acc' : 0,
                'CD_Comp': 0,
                'chamfer-L1': 0,
                'chamfer-L2': 0,
                'F_5': 0, 'P_5': 0, 'R_5': 0,
                'F_10': 0, 'P_10': 0, 'R_10': 0,
                'F_20': 0, 'P_20': 0, 'R_20': 0
            }
            return out_dict

        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )

        chamferL2_old = 0.5 * (completeness + accuracy)
        
        chamfer_l1_val = chamfer_L1(pointcloud, pointcloud_tgt)
        chamfer_l2_val = chamfer_L2(pointcloud, pointcloud_tgt)
        
        

        out_dict = {
            'N_Acc' : accuracy_normals,
            'N_Comp' : completeness_normals,
            'normals': normals_correctness,
            'CD_Acc' : accuracy,
            'CD_Comp': completeness,
            'chamferL2_old': chamferL2_old,
            'chamfer-L1': chamfer_l1_val,
            'chamfer-L2': chamfer_l2_val
        }
        
        for i, tau in enumerate(thresholds):
            P = precision[i]
            R = recall[i]
            F = 0 if (P + R == 0) else 2 * P * R / (P + R)
            
            out_dict[f'P_{tau}'] = P
            out_dict[f'R_{tau}'] = R
            out_dict[f'F_{tau}'] = F
        return out_dict

def chamfer_L1(points_src, points_tgt):
    # X -> Y
    dist_xy, _ = distance_p2p(points_src, None, points_tgt, None)
    # Y -> X
    dist_yx, _ = distance_p2p(points_tgt, None, points_src, None)

    chamfer_l1 = dist_xy.mean() + dist_yx.mean()
    return chamfer_l1

def chamfer_L2(points_src, points_tgt):
    # X -> Y
    dist_xy, _ = distance_p2p(points_src, None, points_tgt, None)
    # Y -> X
    dist_yx, _ = distance_p2p(points_tgt, None, points_src, None)

    chamfer_l2 = (dist_xy ** 2).mean() + (dist_yx ** 2).mean()
    return chamfer_l2


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src, p=2, k=1, workers=8)    # p=2, Euclidean distance

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


def load_ply_points(filename):
    ''' Load points from a PLY file (Point Cloud).
    Args:
        filename (str): input ply file
    '''
    mesh = trimesh.load(filename)
    if hasattr(mesh, 'vertices'):
        points = mesh.vertices
    else:
        points = np.array([])
    
    # Try to get normals if exist
    normals = None
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None: 
        if len(mesh.vertex_normals) == len(points):
            normals = mesh.vertex_normals
        
    return points, normals


def sampleGT(filename, samplepointsnum):
    ''' Sample a mesh, return points with normals.
    Args:
        filename (str): input mesh file
        samplepointsnum (int): numbers of points to be sampled
    '''
    mesh = trimesh.load(filename)
    mesh.fix_normals()  # ensuring outward normals
    sample_random, index_random = trimesh.sample.sample_surface(mesh, samplepointsnum)
    sample_normal = mesh.face_normals[index_random]
    return sample_random, sample_normal