

import os 
import shutil
import trimesh
class utils:

    @staticmethod
    def save_uploaded_model(real_path):
        """Save uploaded model to project's input_model folder"""
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        input_dir = os.path.join(project_root, "input_model")
        os.makedirs(input_dir, exist_ok=True)
        
        # Always use a safe filename to prevent Gradio 404 URL encoding issues (e.g. spaces, Chinese characters)
        original_filename = "current_model.glb"
            
        target_path = os.path.join(input_dir, original_filename)
        shutil.copy2(real_path, target_path)
        return target_path


    @staticmethod
    def load_glb(file_path): 
        try:
            # force='mesh' concatenates all meshes in the scene into a single Trimesh object
            mesh = trimesh.load(file_path, force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                vertices = mesh.vertices.tolist()
                faces = mesh.faces.tolist()
                return vertices, faces
            elif isinstance(mesh, trimesh.Scene):
                # Fallback if force='mesh' didn't perfectly concatenate
                # This usually happens if there are different materials, etc.
                if not mesh.is_empty:
                    concat_mesh = mesh.dump(concatenate=True)
                    return concat_mesh.vertices.tolist(), concat_mesh.faces.tolist()
            return [], []
        except Exception as e:
            print(f"Utils:Error loading GLB: {e}")
            return [], []