import pygame
import numpy as np
import math
import sys
import ctypes
import argparse
import cv2
import time
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import struct

# Physical constants
c = 299792458.0
G = 6.67430e-11

class Camera:
    def __init__(self):
        self.radius = 6.34194e10  # Match original
        self.azimuth = 0.0
        self.elevation = math.pi / 2.0  # Match original
        self.orbit_speed = 0.01
        self.zoom_speed = 2e9
        self.min_radius = 1.5e10
        self.max_radius = 5e12
        self.dragging = False
        self.moving = False
        
        # Animation state for video recording
        self.target_azimuth = 0.0
        self.target_elevation = math.pi / 2.0
        self.target_radius = 6.34194e10
        self.animation_speed = 0.02
    
    def position(self):
        el = max(0.01, min(math.pi - 0.01, self.elevation))
        return np.array([
            self.radius * math.sin(el) * math.cos(self.azimuth),
            self.radius * math.cos(el),
            self.radius * math.sin(el) * math.sin(self.azimuth)
        ], dtype=np.float32)
    
    def target(self):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def update_animation(self):
        """Smoothly animate camera to target position"""
        # Smooth interpolation to target values
        self.azimuth = self.lerp_angle(self.azimuth, self.target_azimuth, self.animation_speed)
        self.elevation = self.lerp(self.elevation, self.target_elevation, self.animation_speed)
        self.radius = self.lerp(self.radius, self.target_radius, self.animation_speed)
        
        # Clamp elevation
        self.elevation = max(0.01, min(math.pi - 0.01, self.elevation))
        self.radius = max(self.min_radius, min(self.max_radius, self.radius))
    
    def lerp(self, a, b, t):
        """Linear interpolation"""
        return a + (b - a) * t
    
    def lerp_angle(self, a, b, t):
        """Angle interpolation handling wrapping"""
        diff = b - a
        # Handle wrapping around 2π
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
        return a + diff * t
    
    def set_target(self, azimuth=None, elevation=None, radius=None):
        """Set target position for smooth animation"""
        if azimuth is not None:
            self.target_azimuth = azimuth
        if elevation is not None:
            self.target_elevation = elevation
        if radius is not None:
            self.target_radius = radius

class BlackHole:
    def __init__(self, mass):
        self.mass = mass
        self.r_s = 2.0 * G * mass / (c * c)

class ObjectData:
    def __init__(self, pos_radius, color, mass):
        self.pos_radius = pos_radius
        self.color = color
        self.mass = mass

class BlackHoleVideoRecorder:
    def __init__(self, width=1920, height=1080, fps=30, duration=30):
        pygame.init()
        
        # Create OpenGL context
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        
        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Black Hole Video Recording")
        
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration
        
        # Higher quality for video
        self.compute_width = width
        self.compute_height = height
        
        # Initialize OpenGL
        glEnable(GL_DEPTH_TEST)
        
        # Physics
        self.camera = Camera()
        self.black_hole = BlackHole(8.54e36)
        
        # Objects (matching original)
        self.objects = [
            ObjectData(
                np.array([4e11, 0.0, 0.0, 4e10], dtype=np.float32),
                np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
                1.98892e30
            ),
            ObjectData(
                np.array([0.0, 0.0, 4e11, 4e10], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                1.98892e30
            ),
            ObjectData(
                np.array([0.0, 0.0, 0.0, self.black_hole.r_s], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                self.black_hole.mass
            )
        ]
        
        # Setup rendering
        self.setup_shaders()
        self.setup_buffers()
        self.setup_fullscreen_quad()
        
        # Video recording setup
        self.setup_video_writer()
        
        # Animation keyframes
        self.setup_animation_sequence()
        
        print(f"🎬 Recording {duration}s video at {width}x{height} @ {fps}fps")
        print(f"📊 Total frames: {self.total_frames}")
    
    def setup_video_writer(self):
        """Setup OpenCV video writer"""
        timestamp = int(time.time())
        self.video_filename = f"blackhole_lensing_{timestamp}.mp4"
        
        # Use high quality codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.video_filename, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
        
        if not self.video_writer.isOpened():
            print("❌ Failed to open video writer")
            sys.exit(1)
    
    def setup_animation_sequence(self):
        """Define cinematic camera movements - consistent fast rotation, with final roll tilt"""
        self.keyframes = [
            # Time (0-1), azimuth, elevation, radius, description
            (0.0, -math.pi/6, math.pi/2 - math.pi/18, 6.34194e10, "Start - slight downward tilt, fast rotation"),
            (0.25, -math.pi/2, math.pi/2 - math.pi/18, 4.5e10, "Continue fast rotation, keep 80°, zoom in"),
            (0.4, -3*math.pi/4, math.pi/2.2, 5e10, "Continue fast rotation, slight downward tilt"),
            (0.6, -5*math.pi/4, math.pi/2.4, 8e10, "Continue fast rotation, zoom out (75°)"),
            (0.8, -3*math.pi/2, math.pi/2.3, 12e10, "Continue fast rotation, zoom out far"),
            (1.0, -2*math.pi, math.pi/2.2, 15e10, "Complete full rotation, slight downward tilt, very far zoom out")
        ]
        
        # Add roll tilt for final dramatic effect
        self.final_roll = math.pi/12  # 15° roll tilt
    
    def get_current_camera_target(self, progress):
        """Interpolate between keyframes based on progress (0-1)"""
        # Find surrounding keyframes
        prev_frame = None
        next_frame = None
        
        for frame in self.keyframes:
            if frame[0] <= progress:
                prev_frame = frame
            else:
                next_frame = frame
                break
        
        if prev_frame is None:
            prev_frame = self.keyframes[0]
        if next_frame is None:
            next_frame = self.keyframes[-1]
        
        # Interpolate between keyframes
        if prev_frame[0] == next_frame[0]:
            t = 0.0
        else:
            t = (progress - prev_frame[0]) / (next_frame[0] - prev_frame[0])
        
        # Smooth interpolation
        t = self.smooth_step(t)
        
        azimuth = self.lerp_angle(prev_frame[1], next_frame[1], t)
        elevation = self.lerp(prev_frame[2], next_frame[2], t)
        radius = self.lerp(prev_frame[3], next_frame[3], t)
        
        return azimuth, elevation, radius
    
    def smooth_step(self, t):
        """Smooth step function for easing"""
        return t * t * (3 - 2 * t)
    
    def lerp(self, a, b, t):
        return a + (b - a) * t
    
    def lerp_angle(self, a, b, t):
        diff = b - a
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
        return a + diff * t
    
    def capture_frame(self):
        """Capture current frame to video"""
        # Read pixels from OpenGL
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and reshape
        image = np.frombuffer(pixels, dtype=np.uint8)
        image = image.reshape((self.height, self.width, 3))
        
        # Flip vertically (OpenGL coordinate system)
        image = np.flip(image, axis=0)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        self.video_writer.write(image_bgr)
    
    def setup_buffers(self):
        # Camera UBO
        self.camera_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.camera_ubo)
        glBufferData(GL_UNIFORM_BUFFER, 128, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.camera_ubo)
        
        # Disk UBO
        self.disk_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.disk_ubo)
        glBufferData(GL_UNIFORM_BUFFER, 8 * 4, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, self.disk_ubo)
        
        # Objects UBO
        self.objects_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.objects_ubo)
        ubo_size = 4 + 12 + 16 * (16 + 16) + 16 * 4
        glBufferData(GL_UNIFORM_BUFFER, ubo_size, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, self.objects_ubo)
    
    def upload_camera_ubo(self):
        cam_pos = self.camera.position()
        target = self.camera.target()
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Apply roll tilt near the end for dramatic effect
        if hasattr(self, 'current_progress') and self.current_progress > 0.7:
            roll_amount = (self.current_progress - 0.7) / 0.3 * self.final_roll
            cos_roll = math.cos(roll_amount)
            sin_roll = math.sin(roll_amount)
            
            # Rotate up and right vectors around forward axis
            new_right = right * cos_roll + up * sin_roll
            new_up = -right * sin_roll + up * cos_roll
            right = new_right / np.linalg.norm(new_right)
            up = new_up / np.linalg.norm(new_up)
        
        data = struct.pack('ffffffffffffffffff?i',
            cam_pos[0], cam_pos[1], cam_pos[2], 0.0,
            right[0], right[1], right[2], 0.0,
            up[0], up[1], up[2], 0.0,
            forward[0], forward[1], forward[2], 0.0,
            math.tan(math.radians(60.0 * 0.5)),
            float(self.width) / float(self.height),
            False,  # Not moving during video recording
            0
        )
        
        glBindBuffer(GL_UNIFORM_BUFFER, self.camera_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def upload_disk_ubo(self):
        r1 = self.black_hole.r_s * 2.2
        r2 = self.black_hole.r_s * 5.2
        num = 2.0
        thickness = 1e9
        
        data = struct.pack('ffff?fff', r1, r2, num, thickness, False, 0.0, 0.0, 0.0)  # Disable grid
        
        glBindBuffer(GL_UNIFORM_BUFFER, self.disk_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def upload_objects_ubo(self):
        count = min(len(self.objects), 16)
        
        data = struct.pack('ifff', count, 0.0, 0.0, 0.0)
        
        for i in range(16):
            if i < count:
                obj = self.objects[i]
                data += struct.pack('ffff', *obj.pos_radius)
            else:
                data += struct.pack('ffff', 0.0, 0.0, 0.0, 0.0)
        
        for i in range(16):
            if i < count:
                obj = self.objects[i]
                data += struct.pack('ffff', *obj.color)
            else:
                data += struct.pack('ffff', 0.0, 0.0, 0.0, 0.0)
        
        for i in range(16):
            if i < count:
                obj = self.objects[i]
                data += struct.pack('f', obj.mass)
            else:
                data += struct.pack('f', 0.0)
        
        glBindBuffer(GL_UNIFORM_BUFFER, self.objects_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def dispatch_compute(self):
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.compute_width, self.compute_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        
        glUseProgram(self.compute_program)
        self.upload_camera_ubo()
        self.upload_disk_ubo()
        self.upload_objects_ubo()
        
        glBindImageTexture(0, self.output_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8)
        
        groups_x = (self.compute_width + 15) // 16
        groups_y = (self.compute_height + 15) // 16
        glDispatchCompute(groups_x, groups_y, 1)
        
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    
    def setup_shaders(self):
        sag_a_rs = self.black_hole.r_s
        
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
        """
        
        fragment_shader = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        }
        """
        
        # Use the same compute shader as the original but with higher quality settings
        compute_shader = f"""
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(binding = 0, rgba8) writeonly uniform image2D outImage;
        layout(std140, binding = 1) uniform Camera {{
            vec3 camPos;     float _pad0;
            vec3 camRight;   float _pad1;
            vec3 camUp;      float _pad2;
            vec3 camForward; float _pad3;
            float tanHalfFov;
            float aspect;
            bool moving;
            int   _pad4;
        }} cam;
        
        layout(std140, binding = 2) uniform Disk {{
            float disk_r1;
            float disk_r2;
            float disk_num;
            float thickness;
            bool enable_grid;
            float _pad0, _pad1, _pad2;
        }};
        
        layout(std140, binding = 3) uniform Objects {{
            int numObjects;
            vec4 objPosRadius[16];
            vec4 objColor[16];
            float  mass[16]; 
        }};
        
        const float SagA_rs = {sag_a_rs:.6e};
        const float D_LAMBDA = 5e6;  // Smaller step for higher quality
        const double ESCAPE_R = 1e30;
        
        vec4 objectColor = vec4(0.0);
        vec3 hitCenter = vec3(0.0);
        float hitRadius = 0.0;
        
        struct Ray {{
            float x, y, z, r, theta, phi;
            float dr, dtheta, dphi;
            float E, L;
        }};
        
        Ray initRay(vec3 pos, vec3 dir) {{
            Ray ray;
            ray.x = pos.x; ray.y = pos.y; ray.z = pos.z;
            ray.r = length(pos);
            ray.theta = acos(pos.z / ray.r);
            ray.phi = atan(pos.y, pos.x);
        
            float dx = dir.x, dy = dir.y, dz = dir.z;
            ray.dr     = sin(ray.theta)*cos(ray.phi)*dx + sin(ray.theta)*sin(ray.phi)*dy + cos(ray.theta)*dz;
            ray.dtheta = (cos(ray.theta)*cos(ray.phi)*dx + cos(ray.theta)*sin(ray.phi)*dy - sin(ray.theta)*dz) / ray.r;
            ray.dphi   = (-sin(ray.phi)*dx + cos(ray.phi)*dy) / (ray.r * sin(ray.theta));
        
            ray.L = ray.r * ray.r * sin(ray.theta) * ray.dphi;
            float f = 1.0 - SagA_rs / ray.r;
            float dt_dL = sqrt((ray.dr*ray.dr)/f + ray.r*ray.r*(ray.dtheta*ray.dtheta + sin(ray.theta)*sin(ray.theta)*ray.dphi*ray.dphi));
            ray.E = f * dt_dL;
        
            return ray;
        }}
        
        bool intercept(Ray ray, float rs) {{
            return ray.r <= rs;
        }}
        
        bool interceptObject(Ray ray) {{
            vec3 P = vec3(ray.x, ray.y, ray.z);
            for (int i = 0; i < numObjects; ++i) {{
                vec3 center = objPosRadius[i].xyz;
                float radius = objPosRadius[i].w;
                if (distance(P, center) <= radius) {{
                    objectColor = objColor[i];
                    hitCenter = center;
                    hitRadius = radius;
                    return true;
                }}
            }}
            return false;
        }}
        
        void geodesicRHS(Ray ray, out vec3 d1, out vec3 d2) {{
            float r = ray.r, theta = ray.theta;
            float dr = ray.dr, dtheta = ray.dtheta, dphi = ray.dphi;
            float f = 1.0 - SagA_rs / r;
            float dt_dL = ray.E / f;
        
            d1 = vec3(dr, dtheta, dphi);
            d2.x = - (SagA_rs / (2.0 * r*r)) * f * dt_dL * dt_dL
                 + (SagA_rs / (2.0 * r*r * f)) * dr * dr
                 + r * (dtheta*dtheta + sin(theta)*sin(theta)*dphi*dphi);
            d2.y = -2.0*dr*dtheta/r + sin(theta)*cos(theta)*dphi*dphi;
            d2.z = -2.0*dr*dphi/r - 2.0*cos(theta)/(sin(theta)) * dtheta * dphi;
        }}
        
        void rk4Step(inout Ray ray, float dL) {{
            vec3 k1a, k1b;
            geodesicRHS(ray, k1a, k1b);
        
            ray.r      += dL * k1a.x;
            ray.theta  += dL * k1a.y;
            ray.phi    += dL * k1a.z;
            ray.dr     += dL * k1b.x;
            ray.dtheta += dL * k1b.y;
            ray.dphi   += dL * k1b.z;
        
            ray.x = ray.r * sin(ray.theta) * cos(ray.phi);
            ray.y = ray.r * sin(ray.theta) * sin(ray.phi);
            ray.z = ray.r * cos(ray.theta);
        }}
        
        bool crossesEquatorialPlane(vec3 oldPos, vec3 newPos) {{
            bool crossed = (oldPos.y * newPos.y < 0.0);
            float r = length(vec2(newPos.x, newPos.z));
            return crossed && (r >= disk_r1 && r <= disk_r2);
        }}
        
        void main() {{
            int WIDTH  = {self.compute_width};
            int HEIGHT = {self.compute_height};
        
            ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
            if (pix.x >= WIDTH || pix.y >= HEIGHT) return;
        
            float u = (2.0 * (pix.x + 0.5) / WIDTH - 1.0) * cam.aspect * cam.tanHalfFov;
            float v = (1.0 - 2.0 * (pix.y + 0.5) / HEIGHT) * cam.tanHalfFov;
            vec3 dir = normalize(u * cam.camRight - v * cam.camUp + cam.camForward);
            Ray ray = initRay(cam.camPos, dir);
        
            vec4 color = vec4(0.0);
            vec3 prevPos = vec3(ray.x, ray.y, ray.z);
            float lambda = 0.0;
        
            bool hitBlackHole = false;
            bool hitDisk      = false;
            bool hitObject    = false;
            
            float closest_approach = ray.r;
            vec3 deflected_direction = dir;
        
            int steps = 80000;  // Higher quality for video
        
            for (int i = 0; i < steps; ++i) {{
                if (intercept(ray, SagA_rs)) {{ hitBlackHole = true; break; }}
                
                closest_approach = min(closest_approach, ray.r);
                
                rk4Step(ray, D_LAMBDA);
                lambda += D_LAMBDA;
        
                vec3 newPos = vec3(ray.x, ray.y, ray.z);
                if (crossesEquatorialPlane(prevPos, newPos)) {{ hitDisk = true; break; }}
                if (interceptObject(ray)) {{ hitObject = true; break; }}
                prevPos = newPos;
                if (ray.r > ESCAPE_R) {{
                    deflected_direction = normalize(newPos - cam.camPos);
                    break;
                }}
            }}
        
            if (hitDisk) {{
                vec3 disk_pos = vec3(ray.x, ray.y, ray.z);
                float disk_radius = length(disk_pos);
                float r_norm = (disk_radius - disk_r1) / (disk_r2 - disk_r1);
                
                float angle = atan(disk_pos.z, disk_pos.x);
                
                float noise1 = sin(angle * 8.0 + r_norm * 12.0) * 0.5 + 0.5;
                float noise2 = sin(angle * 15.0 - r_norm * 8.0 + 2.3) * 0.5 + 0.5;
                float noise3 = sin(angle * 23.0 + r_norm * 20.0 + 4.7) * 0.5 + 0.5;
                
                float edge_variation = mix(noise1, noise2 * noise3, 0.6);
                edge_variation = pow(edge_variation, 1.5);
                
                float disk_density = 1.0 - smoothstep(0.3, 1.2, r_norm + edge_variation * 0.4);
                disk_density *= smoothstep(0.0, 0.15, r_norm);
                
                float structure_noise = 0.7 + 0.3 * mix(noise1, noise2, 0.5);
                disk_density *= structure_noise;
                
                if (disk_density < 0.01) {{
                    vec3 star_dir = normalize(dir);
                    vec2 star_coord = star_dir.xy * 60.0;
                    float star_noise = fract(sin(dot(star_coord, vec2(12.9898, 78.233))) * 43758.5453);
                    
                    vec3 star_color = vec3(0.005, 0.008, 0.015);
                    if (star_noise > 0.998) {{
                        star_color += vec3(0.8, 0.8, 1.0) * (star_noise - 0.998) * 300.0;
                    }}
                    
                    color = vec4(star_color, 1.0);
                }} else {{
                    vec3 hot_core = vec3(1.2, 1.0, 0.9);
                    vec3 medium_temp = vec3(1.0, 0.7, 0.3);
                    vec3 cooler_edge = vec3(0.9, 0.4, 0.15);
                    vec3 cold_outer = vec3(0.6, 0.2, 0.08);
                    
                    vec3 base_color;
                    if (r_norm < 0.25) {{
                        base_color = mix(hot_core, medium_temp, r_norm / 0.25);
                    }} else if (r_norm < 0.6) {{
                        base_color = mix(medium_temp, cooler_edge, (r_norm - 0.25) / 0.35);
                    }} else {{
                        base_color = mix(cooler_edge, cold_outer, (r_norm - 0.6) / 0.4);
                    }}
                    
                    float center_glow = 1.0 / (1.0 + pow(r_norm * 3.5, 1.8));
                    float base_brightness = 0.2 + 0.8 * (1.0 - pow(r_norm, 0.7));
                    float total_brightness = mix(base_brightness, 4.0, center_glow);
                    
                    total_brightness *= disk_density;
                    
                    float micro_turbulence = 0.85 + 0.15 * sin(angle * 11.0 + r_norm * 25.0);
                    
                    vec3 final_color = base_color * total_brightness * micro_turbulence;
                    
                    if (enable_grid && disk_density > 0.3) {{
                        float grid_scale = 10.0;
                        vec2 disk_coords = vec2(disk_pos.x, disk_pos.z) / (disk_r2 * 0.5);
                        float radius_grid = length(disk_coords);
                        float angle_grid = atan(disk_coords.y, disk_coords.x);
                        
                        float radial_lines = abs(sin(angle_grid * grid_scale));
                        float circle_lines = abs(sin(radius_grid * grid_scale * 2.0));
                        float grid_pattern = max(radial_lines, circle_lines);
                        grid_pattern = smoothstep(0.9, 1.0, grid_pattern);
                        
                        vec3 grid_color = base_color * 1.3 + vec3(0.2, 0.15, 0.05);
                        final_color = mix(final_color, grid_color, grid_pattern * 0.25 * disk_density);
                    }}
                    
                    float bloom = center_glow * disk_density * 0.6;
                    final_color += vec3(bloom * 0.9, bloom * 0.7, bloom * 0.3);
                    
                    float redshift_factor = max(0.0, 1.0 - disk_radius / (SagA_rs * 8.0));
                    final_color.r += redshift_factor * 0.15 * disk_density;
                    final_color.gb *= (1.0 - redshift_factor * 0.2);
                    
                    float alpha = min(1.0, disk_density * 2.0);
                    color = vec4(final_color, alpha);
                }}
        
            }} else if (hitBlackHole) {{
                color = vec4(0.0, 0.0, 0.0, 1.0);
        
            }} else if (hitObject) {{
                vec3 P = vec3(ray.x, ray.y, ray.z);
                vec3 N = normalize(P - hitCenter);
                vec3 V = normalize(cam.camPos - P);
                float ambient = 0.1;
                float diff = max(dot(N, V), 0.0);
                float intensity = ambient + (1.0 - ambient) * diff;
                vec3 shaded = objectColor.rgb * intensity;
                color = vec4(shaded, objectColor.a);
        
            }} else {{
                vec3 star_dir = normalize(dir);
                
                vec2 star_coord1 = star_dir.xy * 60.0;
                vec2 star_coord2 = star_dir.xy * 120.0 + vec2(33.7, 17.3);
                vec2 star_coord3 = star_dir.xy * 200.0 + vec2(67.1, 89.2);
                
                float star1 = fract(sin(dot(star_coord1, vec2(12.9898, 78.233))) * 43758.5453);
                float star2 = fract(sin(dot(star_coord2, vec2(39.3467, 91.456))) * 22578.1459);
                float star3 = fract(sin(dot(star_coord3, vec2(67.5324, 45.123))) * 63421.8765);
                
                vec3 star_color = vec3(0.0);
                
                if (star1 > 0.998) {{
                    float brightness = (star1 - 0.998) * 400.0;
                    star_color += vec3(1.0, 0.9, 0.8) * brightness;
                }}
                
                if (star2 > 0.996) {{
                    float brightness = (star2 - 0.996) * 200.0;
                    star_color += vec3(0.8, 0.8, 1.0) * brightness;
                }}
                
                if (star3 > 0.994) {{
                    float brightness = (star3 - 0.994) * 100.0;
                    star_color += vec3(0.6, 0.7, 1.0) * brightness;
                }}
                
                vec3 cosmic_bg = vec3(0.005, 0.008, 0.015);
                
                float nebula = 0.01 * (0.5 + 0.5 * sin(star_dir.x * 2.0)) * (0.5 + 0.5 * sin(star_dir.y * 1.5));
                cosmic_bg += vec3(0.02, 0.01, 0.03) * nebula;
                
                star_color += cosmic_bg;
                
                color = vec4(star_color, 1.0);
            }}
        
            imageStore(outImage, pix, color);
        }}
        """
        
        try:
            self.display_program = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
            self.compute_program = compileProgram(
                compileShader(compute_shader, GL_COMPUTE_SHADER)
            )
        except Exception as e:
            print(f"❌ Shader compilation error: {e}")
            sys.exit(1)
    
    def setup_fullscreen_quad(self):
        quad_vertices = np.array([
            -1.0,  1.0,  0.0, 1.0,
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0
        ], dtype=np.float32)
        
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
        
        self.output_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.compute_width, self.compute_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def render_frame(self):
        """Render a single frame"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Run compute shader
        self.dispatch_compute()
        
        # Display result
        glUseProgram(self.display_program)
        glBindVertexArray(self.quad_vao)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glUniform1i(glGetUniformLocation(self.display_program, "screenTexture"), 0)
        
        glDisable(GL_DEPTH_TEST)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glEnable(GL_DEPTH_TEST)
        
        pygame.display.flip()
    
    def record_video(self):
        """Record the complete video"""
        print("🎬 Starting video recording...")
        start_time = time.time()
        
        for frame_num in range(self.total_frames):
            # Handle pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("❌ Recording interrupted")
                    self.cleanup()
                    return False
            
            # Calculate progress (0-1)
            progress = frame_num / (self.total_frames - 1)
            self.current_progress = progress  # Store for roll effect
            
            # Update camera position based on keyframes
            azimuth, elevation, radius = self.get_current_camera_target(progress)
            self.camera.set_target(azimuth, elevation, radius)
            self.camera.update_animation()
            
            # Render frame
            self.render_frame()
            
            # Capture frame to video
            self.capture_frame()
            
            # Progress update
            if frame_num % (self.fps // 2) == 0:  # Update twice per second
                elapsed = time.time() - start_time
                progress_pct = (frame_num + 1) / self.total_frames * 100
                eta = elapsed / (frame_num + 1) * (self.total_frames - frame_num - 1)
                print(f"📹 Frame {frame_num + 1}/{self.total_frames} ({progress_pct:.1f}%) - ETA: {eta:.1f}s")
        
        total_time = time.time() - start_time
        print(f"✅ Video recording completed in {total_time:.1f}s")
        print(f"💾 Video saved as: {self.video_filename}")
        
        self.cleanup()
        return True
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Black Hole Video Recorder')
    parser.add_argument('--width', type=int, default=1920, help='Video width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Video height (default: 1080)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--duration', type=int, default=30, help='Video duration in seconds (default: 30)')
    parser.add_argument('--resolution', type=str, help='Set resolution (e.g., 1920x1080, 1280x720)')
    
    args = parser.parse_args()
    
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            args.width, args.height = width, height
        except ValueError:
            print("❌ Invalid resolution format. Use format like '1920x1080'")
            sys.exit(1)
    
    print(f"🚀 Initializing Black Hole Video Recorder")
    print(f"📺 Resolution: {args.width}x{args.height}")
    print(f"🎬 FPS: {args.fps}")
    print(f"⏱️  Duration: {args.duration}s")
    
    try:
        recorder = BlackHoleVideoRecorder(
            width=args.width, 
            height=args.height, 
            fps=args.fps, 
            duration=args.duration
        )
        
        success = recorder.record_video()
        
        if success:
            print("🎉 Video recording successful!")
        else:
            print("❌ Video recording failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
