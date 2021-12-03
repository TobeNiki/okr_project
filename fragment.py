import math
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from file import join, make_clean_folder, get_rgbd_file_lists
from os.path import join, exists
"""
シーン再構築システムの最初のステップは、
短いRGBDシーケンスからフラグメントを作成することである。


"""
from optimize_posegraph import optimize_posegraph_for_fragment
from opencv import initialize_opencv
with_opencv = initialize_opencv()
if with_opencv:
    from opencv_pose_estimation import pose_estimation

def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
    """カラー画像とデプス画像からRGBD画像を生成渡す関数"""
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=config["depth_scale"],
        depth_trunc=config["max_depth"],
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    return rgbd_image

def process_single_fragment(fragment_id, color_files, depth_files, n_files,
                            n_fragments, config):
    """
    この関数において入力引数項で述べるカメラ固有行列がセットされ、
    多方向位置合わせ項で述べる位置合わせが行われ
    (RGBD画像ペアの位置合わせ項の処理はこの中で行われる）、
    その結果を受けて(optimize_posegraph.pyで定義されている)
    関数optimize_posegraph_for_fragmentを呼び出し、多方向の位置合わせを行う。
    そしてmake_pointcloud_for_fragment関数によりフラグメントのための点群を生成
    \n
    Parameter:\n
        color_file:カラー画像\n
        depth_files:深度画像\n
        n_files: \n
        n_fragments:\n
        config:設定ファイルのパス\n
            -オプションの引数["path_intrinsic"]がある場合、これはカメラ固有行列
    """
    
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)

    make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config)
    optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    make_pointcloud_for_fragment(config["path_dataset"], color_files,
                                 depth_files, fragment_id, n_fragments,
                                 intrinsic, config)

# examples/Python/ReconstructionSystem/make_fragments.py
def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                           with_opencv, config):
    """
    RGBDイメージのペアをregister_one_rgbd_pair関数で読み込み、
    source_rgbd_imageとtarget_rgbd_imageにセットする。
    Open3D関数のcompute_rgbd_odometryを呼び出してRGBD画像を位置合わせする。
    隣接するRGBD画像の場合、初期値に単位行列を使用する。
    隣接していないRGBD画像に対しては、初期値としてワイド・ベースライン・マッチングを
    使用する。\n
    Parameter:
        s,t:イメージのペアの番号(連番でs<t)\n
        color_file:カラー画像\n
        depth_files:深度画像\n
        intrinsic:カメラ内部行列\n

    """
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], True,
                                        config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], True,
                                        config)

    option = o3d.odometry.OdometryOption()
    option.max_depth_diff = config["max_depth_diff"]
    if abs(s - t) is not 1:
        #隣接していないRGBD画像の場合
        if with_opencv:
            """関数pose_estimationによりOpenCVのORB特徴量を計算し
            ワイド・ベースライン画像に対してスパースな特徴量をマッチさせ、
            次に5ポイントRANSACを実行して大まかなアライメントを推定する。
            そしてこれをcompute_rgbd_odometryの初期値として使用している。"""
            success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                    target_rgbd_image,
                                                    intrinsic, False)
            if success_5pt:
                [success, trans, info] = o3d.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        #隣接するRGBD画像の場合
        odo_init = np.identity(4)
        [success, trans, info] = o3d.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]

# examples/Python/ReconstructionSystem/make_fragments.py
def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config):
    """
    多方向位置合わせ項の手法を使用してシーケンス内のすべてのRGBD画像の
    多方向位置合わせのポーズグラフを作成する
    それぞれのグラフ・ノードは、RGBD画像と、ジオメトリを
    グローバルなフラグメント空間に変換するポーズを表す。
    効率のために、キーフレームだけが使用される。
    ポーズグラフが作成されたならば、(関数process_single_fragmentの中で)
    (optimize_posegraph.pyで定義されている)以下の関数
    optimize_posegraph_for_fragmentを呼び出し、多方向の位置合わせを行う
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(s - sid,
                                                   t - sid,
                                                   trans,
                                                   info,
                                                   uncertain=False))

            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 \
                    and t % config['n_keyframes_per_n_frame'] == 0:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(s - sid,
                                                       t - sid,
                                                       trans,
                                                       info,
                                                       uncertain=True))
    o3d.io.write_pose_graph(
        join(path_dataset, config["template_fragment_posegraph"] % fragment_id),
        pose_graph)

# examples/Python/ReconstructionSystem/make_fragments.py
def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        print(
            "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
            (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))
        rgbd = read_rgbd_image(color_files[i_abs], depth_files[i_abs], False,
                               config)
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config):
    """
    RGBD統合を使用して、それぞれのRGBDシーケンスから色付きのフラグメントを再構築する
    """
    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id, n_fragments,
        join(path_dataset,
             config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name = join(path_dataset,
                    config["template_fragment_pointcloud"] % fragment_id)
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)

def run(config):
    print("making fragments from RGBD sequence.")
    make_clean_folder(join(config["path_dataset"], config["folder_fragment"]))
    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    n_files = len(color_files)
    n_fragments = int(math.ceil(float(n_files) / \
            config['n_frames_per_fragment']))

    if config["python_multi_threading"]:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(), n_fragments)
        Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(
            fragment_id, color_files, depth_files, n_files, n_fragments, config)
                                    for fragment_id in range(n_fragments))
    else:
        for fragment_id in range(n_fragments):
            process_single_fragment(fragment_id, color_files, depth_files,
                                    n_files, n_fragments, config)

from os import listdir, makedirs
from os.path import exists, isfile, join, splitext, dirname, basename
import shutil
import re
from warnings import warn
import json
import open3d as o3d


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if exists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
            return path
    raise FileNotFoundError(
        f"None of the folders {folder_names} found in {path_dataset}")


def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    return path_color, path_depth


def get_rgbd_file_lists(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + \
            get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        shutil.rmtree(path_folder)
        makedirs(path_folder)


def check_folder_structure(path_dataset):
    if isfile(path_dataset) and path_dataset.endswith(".bag"):
        return
    path_color, path_depth = get_rgbd_folders(path_dataset)
    assert exists(path_depth), \
            "Path %s is not exist!" % path_depth
    assert exists(path_color), \
            "Path %s is not exist!" % path_color


def write_poses_to_log(filename, poses):
    with open(filename, 'w') as f:
        for i, pose in enumerate(poses):
            f.write('{} {} {}\n'.format(i, i, i + 1))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[3, 0], pose[3, 1], pose[3, 2], pose[3, 3]))


def read_poses_from_log(traj_log):
    import numpy as np

    trans_arr = []
    with open(traj_log) as f:
        content = f.readlines()

        # Load .log file.
        for i in range(0, len(content), 5):
            # format %d (src) %d (tgt) %f (fitness)
            data = list(map(float, content[i].strip().split(' ')))
            ids = (int(data[0]), int(data[1]))
            fitness = data[2]

            # format %f x 16
            T_gt = np.array(
                list(map(float, (''.join(
                    content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

            trans_arr.append(T_gt)

    return trans_arr


def extract_rgbd_frames(rgbd_video_file):
    """
    Extract color and aligned depth frames and intrinsic calibration from an
    RGBD video file (currently only RealSense bag files supported). Folder
    structure is:
        <directory of rgbd_video_file/<rgbd_video_file name without extension>/
            {depth/00000.jpg,color/00000.png,intrinsic.json}
    """
    frames_folder = join(dirname(rgbd_video_file),
                         basename(splitext(rgbd_video_file)[0]))
    path_intrinsic = join(frames_folder, "intrinsic.json")
    if isfile(path_intrinsic):
        warn(f"Skipping frame extraction for {rgbd_video_file} since files are"
             " present.")
    else:
        rgbd_video = o3d.t.io.RGBDVideoReader.create(rgbd_video_file)
        rgbd_video.save_frames(frames_folder)
    with open(path_intrinsic) as intr_file:
        intr = json.load(intr_file)
    depth_scale = intr["depth_scale"]
    return frames_folder, path_intrinsic, depth_scale
