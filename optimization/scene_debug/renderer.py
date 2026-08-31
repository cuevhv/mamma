import colorsys
import os
import platform

# Set OpenGL backend before importing pyrender so cluster jobs can override it
# via environment variables (e.g., PYOPENGL_PLATFORM=osmesa).
# if 'PYOPENGL_PLATFORM' not in os.environ:
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import pyrender
import trimesh

OPACITY = 0.6


class Renderer:
    def __init__(self, focal_length_px, img_w, img_h, principal_p_x, principal_p_y, faces,
                 focal_length_px_y=None):

        try:
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=img_w,
                viewport_height=img_h,
                point_size=1.0,
            )
        except Exception as e:
            ogl_backend = os.environ.get("PYOPENGL_PLATFORM", "<unset>")
            egl_device_id = os.environ.get("EGL_DEVICE_ID", "<unset>")
            raise RuntimeError(
                "Failed to initialize pyrender offscreen renderer. "
                f"PYOPENGL_PLATFORM={ogl_backend}, "
                f"EGL_DEVICE_ID={egl_device_id}. "
                "On clusters, ensure EGL is available for your allocated GPU "
                "or set PYOPENGL_PLATFORM=osmesa if OSMesa is installed."
            ) from e
        self.camera_center = [principal_p_x, principal_p_y]
        self.focal_length = focal_length_px
        self.focal_length_y = focal_length_px if focal_length_px_y is None else focal_length_px_y
        self.faces = faces

    def render_front_view(self, verts, bg_img_bgr=None, bg_color=(0, 0, 0, 0), return_depth=False):
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        camera = pyrender.camera.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length_y,
            cx=self.camera_center[0],
            cy=self.camera_center[1],
        )
        scene.add(camera, pose=np.eye(4))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0]))
        scene.add(light, pose=trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0]))

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        num_people = len(verts)

        for person_idx in range(num_people):
            mesh = trimesh.Trimesh(verts[person_idx], self.faces, process=False)
            mesh.apply_transform(rot)
            mesh_color = colorsys.hsv_to_rgb(float(person_idx) / max(num_people, 1), 0.5, 1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode="OPAQUE",
                baseColorFactor=(mesh_color[0], mesh_color[1], mesh_color[2], 1.0),
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh)

        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        render_rgb = color_rgba[:, :, :3].astype(np.float32)

        if bg_img_bgr is None:
            out_img = color_rgba[:, :, :3][:, :, ::-1]
        else:
            bg_img_bgr = bg_img_bgr.astype(np.float32)
            render_bgr = render_rgb[:, :, ::-1]
            alpha = (color_rgba[:, :, 3:4].astype(np.float32) / 255.0) * OPACITY
            out_img = render_bgr * alpha + bg_img_bgr * (1.0 - alpha)
            out_img = out_img.astype(np.uint8)

        if return_depth:
            return out_img, depth_map
        return out_img

    def delete(self):
        self.renderer.delete()
