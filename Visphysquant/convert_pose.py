import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import zipfile
import shutil
import argparse
import cv2
import liblzfse
import glob

def load_depth(filepath, is_portrait=False):
    """Read a `.depth` file, automatically infer its shape, and return it."""
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    # Expected size candidates (Short, Long)
    candidates = {
        640 * 480: (480, 640),
        256 * 192: (192, 256)
    }

    if depth_img.size in candidates:
        short_edge, long_edge = candidates[depth_img.size]
        
        if is_portrait:
            # Portrait: (Height, Width) = (Long, Short) -> e.g. 640x480, 256x192
            depth_img = depth_img.reshape((long_edge, short_edge))
        else:
            # Landscape: (Height, Width) = (Short, Long) -> e.g. 480x640, 192x256
            depth_img = depth_img.reshape((short_edge, long_edge))
    else:
        # Other sizes (warn and assume square, or handle as an error)
        side = int(np.sqrt(depth_img.size))
        depth_img = depth_img.reshape((side, side))
        print(f"Warning: Unknown depth size {depth_img.size}, reshaped to {side}x{side}")

    return depth_img

def unzip_r3d(r3d_file, extract_dir):
    """Extract a `.r3d` file."""
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)
    
    with zipfile.ZipFile(r3d_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted {r3d_file} to {extract_dir}")

def process_record3d_data(input_json_path, output_pose_txt, output_meta_json, target_size=None):
    # 1. Load the JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # --- Part 1: Pose conversion (with normalization) ---
    poses = data.get("poses", [])
    output_lines = []
    
    if poses:
        first_pose_inv = None
        for frame_id, pose in enumerate(poses):
            # Record3D format: [tx, ty, tz, qx, qy, qz, qw]
            tx, ty, tz = pose[0], pose[1], pose[2]
            qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
            
            # Quaternion -> rotation matrix
            rotation = R.from_quat([qx, qy, qz, qw])
            rot_matrix = rotation.as_matrix()
            
            # Create a 4x4 matrix
            current_matrix = np.eye(4)
            current_matrix[:3, :3] = rot_matrix
            current_matrix[:3, 3] = [tx, ty, tz]
            
            # Normalization: multiply by the inverse of the first frame
            if frame_id == 0:
                first_pose_inv = np.linalg.inv(current_matrix)
                final_matrix = np.eye(4) # The first frame is always the identity matrix
            else:
                final_matrix = first_pose_inv @ current_matrix

            # Convert to a single-line string
            flat_matrix = final_matrix.flatten()
            matrix_str = " ".join(map(str, flat_matrix))
            line = f"{frame_id} {matrix_str}"
            output_lines.append(line)

        # Write `cam_poses.txt`
        with open(output_pose_txt, 'w') as f:
            f.write("\n".join(output_lines))
        print(f"Pose data saved to: {output_pose_txt}")

    # --- Part 2: Metadata extraction and correction ---
    metadata_out = {}
    
    # Get the original resolution
    w_orig = data.get("w", 640)
    h_orig = data.get("h", 480)
    
    # If a target size (depth size) is specified, compute the scaling
    scale_x = 1.0
    scale_y = 1.0
    if target_size is not None:
        target_w, target_h = target_size
        scale_x = target_w / w_orig
        scale_y = target_h / h_orig
        metadata_out["w"] = target_w
        metadata_out["h"] = target_h
    else:
        metadata_out["w"] = w_orig
        metadata_out["h"] = h_orig

    # Process K (intrinsics) and apply scaling correction
    if "K" in data:
        # When K already exists (usually a flat list)
        # Be careful with Record3D's storage order
        pass # Strongly implementation-dependent, so using `perFrameIntrinsicCoeffs` below is recommended
    
    # Record3D usually provides `perFrameIntrinsicCoeffs`
    if "perFrameIntrinsicCoeffs" in data and len(data["perFrameIntrinsicCoeffs"]) > 0:
        coeffs = data["perFrameIntrinsicCoeffs"][0]
        # [fx, fy, cx, cy]
        fx, fy, cx, cy = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        
        # Apply scaling
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        
        # 3x3 matrix (flat, column-major or row-major for Open3D)
        # Open3D's PinholeCameraIntrinsic expects [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        metadata_out["K"] = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    # Write metadata JSON
    with open(output_meta_json, 'w') as f:
        json.dump(metadata_out, f, indent=4)
    print(f"Metadata saved to: {output_meta_json}")

def process_single_r3d(r3d_file, output_root):
    # Use the `.r3d` filename (without extension) as the extraction directory name
    r3d_filename = os.path.splitext(os.path.basename(r3d_file))[0]
    extract_path = os.path.join(output_root, r3d_filename)
    
    # Check whether it has already been processed (skip if `cam_poses.txt` and `camera_intrinsics.json` exist)
    pose_output_check = os.path.join(extract_path, "cam_poses.txt")
    meta_output_check = os.path.join(extract_path, "camera_intrinsics.json")
    
    if os.path.exists(pose_output_check) and os.path.exists(meta_output_check):
        print(f"Skipping {r3d_filename}: Already processed.")
        return

    # Extract the `.r3d` file
    unzip_r3d(r3d_file, extract_path)
    
    rgbd_path = os.path.join(extract_path, "rgbd")
    rgb_out_dir = os.path.join(extract_path, "rgb")
    depth_out_dir = os.path.join(extract_path, "depth")

    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(depth_out_dir, exist_ok=True)

    if not os.path.exists(rgbd_path):
        print(f"Error: {rgbd_path} not found.")
        return

    # --- Step 1: Determine image orientation (Portrait/Landscape) from the RGB image ---
    jpg_files = [f for f in os.listdir(rgbd_path) if f.endswith('.jpg')]
    is_portrait = False
    
    if jpg_files:
        # Read the first image and check its shape
        sample_img_path = os.path.join(rgbd_path, jpg_files[0])
        sample_img = cv2.imread(sample_img_path)
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            if h > w:
                is_portrait = True
            print(f"Detected RGB Orientation: {'Portrait' if is_portrait else 'Landscape'} (RGB: {w}x{h})")
        else:
            print(f"Warning: Failed to read {sample_img_path} for orientation check.")
    else:
        print("Warning: No jpg files found to determine orientation. Assuming Landscape.")

    # --- Step 2: Read one depth file and determine the target resolution ---
    depth_files = [f for f in os.listdir(rgbd_path) if f.endswith('.depth')]
    if not depth_files:
        print("Error: No depth files found.")
        return
    
    sample_depth = load_depth(os.path.join(rgbd_path, depth_files[0]), is_portrait=is_portrait)
    target_h, target_w = sample_depth.shape
    print(f"Detected Depth Resolution: {target_w}x{target_h}")

    # --- Step 2: Process metadata (pass resolution info and correct K) ---
    metadata_path = os.path.join(extract_path, "metadata")
    if os.path.exists(metadata_path):
        pose_output = os.path.join(extract_path, "cam_poses.txt")
        meta_output = os.path.join(extract_path, "camera_intrinsics.json")
        process_record3d_data(metadata_path, pose_output, meta_output, target_size=(target_w, target_h))
    else:
        print(f"Error: {metadata_path} not found.")

    # --- Step 3: Image conversion processing ---
    depth_mismatch_frames = []
    for f in sorted(os.listdir(rgbd_path)):
        frame_id, ext = os.path.splitext(f)
        src_path = os.path.join(rgbd_path, f)

        if ext == '.jpg':
            img = cv2.imread(src_path)
            if img is not None:
                # [Important] Resize RGB to match the depth size
                #if (img.shape[1] != target_w) or (img.shape[0] != target_h):
                #    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(rgb_out_dir, f"{frame_id}.jpg"), img)

        elif ext == '.depth':
            depth_img = load_depth(src_path, is_portrait=is_portrait)
            
            # [Important] Convert to uint16 (millimeters) to preserve metric information
            # The original normalize(0-255) code has been removed
            depth_mm = (depth_img * 1000).astype(np.uint16)
            
            cv2.imwrite(os.path.join(depth_out_dir, f"{frame_id}.png"), depth_mm)

            total_pixels = int(depth_mm.size)
            nonzero_pixels = int(np.count_nonzero(depth_mm))
            if total_pixels != nonzero_pixels:
                depth_mismatch_frames.append(
                    f"{r3d_filename}/{frame_id} nonzero_pixels={nonzero_pixels}"
                )
        
    print(f"RGB images saved to: {rgb_out_dir}")
    print(f"Depth images saved to: {depth_out_dir}")
    if depth_mismatch_frames:
        print("Depth check (total_pixels != nonzero_pixels) detected in:")
        for name in depth_mismatch_frames:
            print(name)

def is_r3d_like_file(filename):
    """Determine whether the filename is `.r3d`-like, including names such as '.r3d copy'."""
    return ".r3d" in filename

def main():
    # Base directory to search
    input_base_dir = './r3d_data'
    # Root output directory
    output_root_dir = './processed_data'
    
    # Create the output directory if it does not exist
    os.makedirs(output_root_dir, exist_ok=True)
    
    # Search for all `.r3d` files under the base directory (including names like '.r3d copy')
    r3d_files = []
    for root, _, files in os.walk(input_base_dir):
        for file in files:
            if is_r3d_like_file(file):
                r3d_files.append(os.path.join(root, file))
    
    print(f"Found {len(r3d_files)} .r3d files.")
    
    for r3d_file in r3d_files:
        print(f"Processing: {r3d_file}")
        try:
            process_single_r3d(r3d_file, output_root_dir)
        except Exception as e:
            print(f"Failed to process {r3d_file}: {e}")

if __name__ == '__main__':
    main()
