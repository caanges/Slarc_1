import trimesh
import numpy as np

# 1. Load your cleaned mesh
filename = 'ugv_clean.obj'
try:
    mesh = trimesh.load(filename)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

print(f"Mesh loaded: {len(mesh.vertices)} vertices.")

# 2. Alternative Salience: Vertex Defect (Corner Detection)
# This is much more stable across different versions of trimesh
print("Analyzing geometry for sharp features...")
try:
    # This finds 'Pointy' areas (corners)
    salience = np.abs(trimesh.curvature.vertex_defects(mesh))
except:
    # If all else fails, use a simple distance-from-center weight
    print("Curvature failed, using fallback spread...")
    salience = np.ones(len(mesh.vertices))

# 3. Weighted Farthest Point Sampling
def get_keypoints(pts, scores, n):
    selected_indices = [np.argmax(scores)]
    farthest_pts = [pts[selected_indices[0]]]
    distances = np.full(len(pts), np.inf)
    
    for _ in range(1, n):
        last_pt = farthest_pts[-1]
        dist_to_last = np.linalg.norm(pts - last_pt, axis=1)
        distances = np.minimum(distances, dist_to_last)
        
        # Multiply distance by salience (sharpness)
        # This forces the points to snap to corners
        combined_score = distances * scores
        next_idx = np.argmax(combined_score)
        farthest_pts.append(pts[next_idx])
        
    return np.array(farthest_pts)

# Extract 5 points
keypoints = get_keypoints(mesh.vertices, salience, 5)

print("\n" + "="*50)
print("  SUCCESS: KEYPOINTS GENERATED")
print("="*50)
for i, p in enumerate(keypoints):
    print(f"Point {i+1}: X={p[0]:.6f}, Y={p[1]:.6f}, Z={p[2]:.6f}")
print("="*50)