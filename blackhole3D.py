import pygame
import numpy as np
import math
import sys
import ctypes
import argparse
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import struct

# Physical constants
c = 299792458.0
G = 6.67430e-11

class Camera:
    def __init__(self):
        self.radius = 6.34194e10  # Match C++ reference
        self.azimuth = 0.0
        self.elevation = math.pi / 2.0  # Match C++ reference
        self.orbit_speed = 0.01
        self.zoom_speed = 2e9  # More sensitive zoom
        self.min_radius = 1.5e10  # Closer minimum (1.5x Schwarzschild radius)
        self.max_radius = 5e12    # Further maximum for wider view
        self.dragging = False
        self.moving = False
    
    def position(self):
        # Match C++ clamping and coordinate system
        el = max(0.01, min(math.pi - 0.01, self.elevation))
        return np.array([
            self.radius * math.sin(el) * math.cos(self.azimuth),
            self.radius * math.cos(el),
            self.radius * math.sin(el) * math.sin(self.azimuth)
        ], dtype=np.float32)
    
    def target(self):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def update(self):
        # Update moving state for compute shader optimization
        pass

class BlackHole:
    def __init__(self, mass):
        self.mass = mass
        self.r_s = 2.0 * G * mass / (c * c)

class ObjectData:
    def __init__(self, pos_radius, color, mass):
        self.pos_radius = pos_radius  # vec4: xyz=position, w=radius
        self.color = color            # vec4: rgba
        self.mass = mass              # float

class BlackHoleSimulation:
    def __init__(self, width=720, height=480, enable_grid=False):
        pygame.init()
        
        # Create OpenGL context with compute shader support
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        
        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Black Hole Simulation with Gravitational Lensing")
        
        self.width = width
        self.height = height
        self.enable_grid = enable_grid
        
        # Higher compute resolution for better quality
        # Scale more aggressively for high resolution displays
        scale_factor = min(0.6, max(0.3, 1200 / width))  # Between 30% and 60% of window size
        self.compute_width = max(400, int(width * scale_factor))
        self.compute_height = max(300, int(height * scale_factor))
        
        # Initialize OpenGL
        glEnable(GL_DEPTH_TEST)
        
        # Physics
        self.camera = Camera()
        self.black_hole = BlackHole(8.54e36)  # Sagittarius A* mass
        
        # Objects in scene (matching C++ reference)
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
        
        self.clock = pygame.time.Clock()
        
    def setup_buffers(self):
        # Create uniform buffer objects matching C++ reference
        
        # Camera UBO
        self.camera_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.camera_ubo)
        glBufferData(GL_UNIFORM_BUFFER, 128, None, GL_DYNAMIC_DRAW)  # ~128 bytes
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.camera_ubo)  # binding = 1
        
        # Disk UBO (4 floats + 1 bool + 3 float padding = 8 floats total)
        self.disk_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.disk_ubo)
        glBufferData(GL_UNIFORM_BUFFER, 8 * 4, None, GL_DYNAMIC_DRAW)  # 8 floats
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, self.disk_ubo)  # binding = 2
        
        # Objects UBO
        self.objects_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.objects_ubo)
        # Size: int + padding + 16*(vec4 + vec4) + 16*float
        ubo_size = 4 + 12 + 16 * (16 + 16) + 16 * 4  # int + padding + 16*(vec4+vec4) + 16*float
        glBufferData(GL_UNIFORM_BUFFER, ubo_size, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, self.objects_ubo)  # binding = 3
        
        
        # Initialize grid
        self.setup_grid()
    
    def setup_grid(self):
        """Setup spacetime grid VAO/VBO"""
        self.grid_vao = glGenVertexArrays(1)
        self.grid_vbo = glGenBuffers(1)
        self.grid_ebo = glGenBuffers(1)
        self.grid_index_count = 0
        
        # Generate initial grid
        self.generate_grid()
    
    def generate_grid(self):
        """Generate warped spacetime grid (matching C++ reference)"""
        grid_size = 20  # Smaller grid for better performance
        spacing = 2e10  # Larger spacing for better visibility
        
        vertices = []
        indices = []
        
        # Generate grid vertices
        for z in range(grid_size + 1):
            for x in range(grid_size + 1):
                world_x = (x - grid_size // 2) * spacing
                world_z = (z - grid_size // 2) * spacing
                
                y = -1e10  # Start below equatorial plane for better visibility
                
                # Only warp around the central black hole (first object)
                if len(self.objects) > 0:
                    obj = self.objects[2]  # The black hole object
                    obj_pos = obj.pos_radius[:3]
                    mass = obj.mass
                    
                    r_s = self.black_hole.r_s
                    dx = world_x - obj_pos[0]
                    dz = world_z - obj_pos[2]
                    dist = math.sqrt(dx * dx + dz * dz)
                    
                    # Create visible curvature
                    if dist > r_s * 1.1:  # Safety margin
                        # Smooth curvature function
                        warp_factor = r_s / dist
                        delta_y = warp_factor * warp_factor * 5e10  # Amplify the effect
                        y -= delta_y  # Negative to create a well
                    else:
                        # Deep pit near black hole
                        y -= 8e10
                
                vertices.extend([world_x, y, world_z])
        
        # Generate grid indices for lines
        for z in range(grid_size):
            for x in range(grid_size):
                i = z * (grid_size + 1) + x
                
                # Horizontal line
                indices.extend([i, i + 1])
                
                # Vertical line
                indices.extend([i, i + grid_size + 1])
        
        # Upload to GPU
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        glBindVertexArray(self.grid_vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.grid_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        
        self.grid_index_count = len(indices)
        
        glBindVertexArray(0)
    
    def draw_grid(self, view_proj_matrix):
        """Draw the spacetime grid"""
        glUseProgram(self.grid_program)
        
        # Set view-projection matrix
        glUniformMatrix4fv(
            glGetUniformLocation(self.grid_program, "viewProj"),
            1, GL_FALSE, view_proj_matrix.T  # Transpose for OpenGL column-major
        )
        
        glBindVertexArray(self.grid_vao)
        
        # Render grid with proper blending and depth testing
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)  # Allow grid to render at same depth as background
        
        # Set line width for better visibility
        glLineWidth(1.5)
        
        # Draw grid lines
        glDrawElements(GL_LINES, self.grid_index_count, GL_UNSIGNED_INT, None)
        
        # Restore state
        glLineWidth(1.0)
        glDepthFunc(GL_LESS)
        glBindVertexArray(0)
        glDisable(GL_BLEND)
    
    def upload_camera_ubo(self):
        """Upload camera data to uniform buffer"""
        # Calculate camera vectors
        cam_pos = self.camera.position()
        target = self.camera.target()
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Pack data according to std140 layout (matching shader uniform block)
        data = struct.pack('ffffffffffffffffff?i',
            cam_pos[0], cam_pos[1], cam_pos[2], 0.0,     # vec3 + padding
            right[0], right[1], right[2], 0.0,           # vec3 + padding
            up[0], up[1], up[2], 0.0,                    # vec3 + padding
            forward[0], forward[1], forward[2], 0.0,     # vec3 + padding
            math.tan(math.radians(60.0 * 0.5)),          # tanHalfFov
            float(self.width) / float(self.height),      # aspect
            self.camera.moving,                          # moving (bool)
            0                                            # padding (int)
        )
        
        glBindBuffer(GL_UNIFORM_BUFFER, self.camera_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def upload_disk_ubo(self):
        """Upload disk parameters to uniform buffer"""
        r1 = self.black_hole.r_s * 2.2  # Inner radius
        r2 = self.black_hole.r_s * 5.2  # Outer radius
        num = 2.0                       # Number of rays
        thickness = 1e9                 # Thickness
        
        # Pack with grid enable flag and padding
        data = struct.pack('ffff?fff', r1, r2, num, thickness, self.enable_grid, 0.0, 0.0, 0.0)
        
        glBindBuffer(GL_UNIFORM_BUFFER, self.disk_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def upload_objects_ubo(self):
        """Upload objects data to uniform buffer"""
        count = min(len(self.objects), 16)
        
        # Pack number of objects + padding
        data = struct.pack('ifff', count, 0.0, 0.0, 0.0)
        
        # Pack position/radius data (16 vec4s)
        for i in range(16):
            if i < count:
                obj = self.objects[i]
                data += struct.pack('ffff', *obj.pos_radius)
            else:
                data += struct.pack('ffff', 0.0, 0.0, 0.0, 0.0)
        
        # Pack color data (16 vec4s)
        for i in range(16):
            if i < count:
                obj = self.objects[i]
                data += struct.pack('ffff', *obj.color)
            else:
                data += struct.pack('ffff', 0.0, 0.0, 0.0, 0.0)
        
        # Pack mass data (16 floats)
        for i in range(16):
            if i < count:
                obj = self.objects[i]
                data += struct.pack('f', obj.mass)
            else:
                data += struct.pack('f', 0.0)
        
        glBindBuffer(GL_UNIFORM_BUFFER, self.objects_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def dispatch_compute(self):
        """Dispatch compute shader for ray tracing"""
        # Use consistent resolution - no more switching when moving
        cw = self.compute_width
        ch = self.compute_height
        
        # Reallocate texture if needed
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cw, ch, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        
        # Use compute program and upload UBOs
        glUseProgram(self.compute_program)
        self.upload_camera_ubo()
        self.upload_disk_ubo()
        self.upload_objects_ubo()
        
        # Bind texture as image unit 0
        glBindImageTexture(0, self.output_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8)
        
        # Dispatch compute groups
        groups_x = (cw + 15) // 16  # Round up division
        groups_y = (ch + 15) // 16
        glDispatchCompute(groups_x, groups_y, 1)
        
        # Memory barrier
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    
    def setup_shaders(self):
        # Calculate Schwarzschild radius for shader
        sag_a_rs = self.black_hole.r_s
        
        # Vertex shader for fullscreen quad
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
        
        # Fragment shader for displaying compute shader results
        fragment_shader = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        }
        """
        
        # Compute shader for geodesic ray tracing (matching C++ reference)
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
        const float D_LAMBDA = 1e7;
        const double ESCAPE_R = 1e30;
        
        // Globals to store hit info
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
        
            // Init Ray
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
            
            // Track closest approach to black hole for lensing effects
            float closest_approach = ray.r;
            vec3 deflected_direction = dir;  // Track how much the ray was deflected
        
             int steps = 60000;  // Consistent high quality
        
            for (int i = 0; i < steps; ++i) {{
                if (intercept(ray, SagA_rs)) {{ hitBlackHole = true; break; }}
                
                // Track closest approach for lensing calculations
                closest_approach = min(closest_approach, ray.r);
                
                rk4Step(ray, D_LAMBDA);
                lambda += D_LAMBDA;
        
                vec3 newPos = vec3(ray.x, ray.y, ray.z);
                if (crossesEquatorialPlane(prevPos, newPos)) {{ hitDisk = true; break; }}
                if (interceptObject(ray)) {{ hitObject = true; break; }}
                prevPos = newPos;
                if (ray.r > ESCAPE_R) {{
                    // Calculate final deflected direction
                    deflected_direction = normalize(newPos - cam.camPos);
                    break;
                }}
            }}
        
             if (hitDisk) {{
                 vec3 disk_pos = vec3(ray.x, ray.y, ray.z);
                 float disk_radius = length(disk_pos);
                 float r_norm = (disk_radius - disk_r1) / (disk_r2 - disk_r1);
                 
                 // Interstellar-style irregular edge density with wispy falloff
                 float angle = atan(disk_pos.z, disk_pos.x);
                 
                 // Multiple noise layers for complex, irregular edge structure
                 float noise1 = sin(angle * 8.0 + r_norm * 12.0) * 0.5 + 0.5;
                 float noise2 = sin(angle * 15.0 - r_norm * 8.0 + 2.3) * 0.5 + 0.5;
                 float noise3 = sin(angle * 23.0 + r_norm * 20.0 + 4.7) * 0.5 + 0.5;
                 
                 // Combine noises for wispy, irregular structure
                 float edge_variation = mix(noise1, noise2 * noise3, 0.6);
                 edge_variation = pow(edge_variation, 1.5); // Sharpen the wisps
                 
                 // Create soft, irregular boundary - no hard edge!
                 float disk_density = 1.0 - smoothstep(0.3, 1.2, r_norm + edge_variation * 0.4);
                 disk_density *= smoothstep(0.0, 0.15, r_norm); // Fade in from center
                 
                 // Apply irregular structure throughout the disk
                 float structure_noise = 0.7 + 0.3 * mix(noise1, noise2, 0.5);
                 disk_density *= structure_noise;
                 
                 // Only render if there's significant density
                 if (disk_density < 0.01) {{
                     // Treat as background space instead of disk
            vec3 star_dir = normalize(dir);
                     vec2 star_coord = star_dir.xy * 60.0;
            float star_noise = fract(sin(dot(star_coord, vec2(12.9898, 78.233))) * 43758.5453);
            
                     vec3 star_color = vec3(0.005, 0.008, 0.015);
                     if (star_noise > 0.998) {{
                         star_color += vec3(0.8, 0.8, 1.0) * (star_noise - 0.998) * 300.0;
                     }}
                     
                     color = vec4(star_color, 1.0);
                 }} else {{
                     // Realistic accretion disk colors with Interstellar-style warmth
                     vec3 hot_core = vec3(1.2, 1.0, 0.9);      // Warm white-hot center
                     vec3 medium_temp = vec3(1.0, 0.7, 0.3);   // Warm orange
                     vec3 cooler_edge = vec3(0.9, 0.4, 0.15);  // Warm red-orange
                     vec3 cold_outer = vec3(0.6, 0.2, 0.08);   // Deep red-brown
                     
                     // Smooth color transitions
                     vec3 base_color;
                     if (r_norm < 0.25) {{
                         base_color = mix(hot_core, medium_temp, r_norm / 0.25);
                     }} else if (r_norm < 0.6) {{
                         base_color = mix(medium_temp, cooler_edge, (r_norm - 0.25) / 0.35);
                     }} else {{
                         base_color = mix(cooler_edge, cold_outer, (r_norm - 0.6) / 0.4);
                     }}
                     
                     // Brightness with dramatic center glow
                     float center_glow = 1.0 / (1.0 + pow(r_norm * 3.5, 1.8));
                     float base_brightness = 0.2 + 0.8 * (1.0 - pow(r_norm, 0.7));
                     float total_brightness = mix(base_brightness, 4.0, center_glow);
                     
                     // Apply density for wispy edges
                     total_brightness *= disk_density;
                     
                     // Subtle turbulence for organic motion
                     float micro_turbulence = 0.85 + 0.15 * sin(angle * 11.0 + r_norm * 25.0);
                     
                     vec3 final_color = base_color * total_brightness * micro_turbulence;
                     
                     // Grid pattern (much more subtle on wispy disk)
                     if (enable_grid && disk_density > 0.3) {{
                         float grid_scale = 10.0;
                         vec2 disk_coords = vec2(disk_pos.x, disk_pos.z) / (disk_r2 * 0.5);
                         float radius_grid = length(disk_coords);
                         float angle_grid = atan(disk_coords.y, disk_coords.x);
                         
                         float radial_lines = abs(sin(angle_grid * grid_scale));
                         float circle_lines = abs(sin(radius_grid * grid_scale * 2.0));
                         float grid_pattern = max(radial_lines, circle_lines);
                         grid_pattern = smoothstep(0.9, 1.0, grid_pattern);
                         
                         // Very subtle grid that doesn't break the cinematic look
                         vec3 grid_color = base_color * 1.3 + vec3(0.2, 0.15, 0.05);
                         final_color = mix(final_color, grid_color, grid_pattern * 0.25 * disk_density);
                     }}
                     
                     // Atmospheric bloom for that cinematic glow
                     float bloom = center_glow * disk_density * 0.6;
                     final_color += vec3(bloom * 0.9, bloom * 0.7, bloom * 0.3);
                     
                     // Gravitational redshift (subtle)
                     float redshift_factor = max(0.0, 1.0 - disk_radius / (SagA_rs * 8.0));
                     final_color.r += redshift_factor * 0.15 * disk_density;
                     final_color.gb *= (1.0 - redshift_factor * 0.2);
                     
                     // Alpha blending for soft edges
                     float alpha = min(1.0, disk_density * 2.0);
                     color = vec4(final_color, alpha);
                 }}
        
            }} else if (hitBlackHole) {{
                color = vec4(0.0, 0.0, 0.0, 1.0);
        
            }} else if (hitObject) {{
                // Compute shading
                vec3 P = vec3(ray.x, ray.y, ray.z);
                vec3 N = normalize(P - hitCenter);
                vec3 V = normalize(cam.camPos - P);
                float ambient = 0.1;
                float diff = max(dot(N, V), 0.0);
                float intensity = ambient + (1.0 - ambient) * diff;
                vec3 shaded = objectColor.rgb * intensity;
                color = vec4(shaded, objectColor.a);
        
            }} else {{
                // Enhanced starfield background with better contrast
            vec3 star_dir = normalize(dir);
                
                // Multiple star layers for depth
                vec2 star_coord1 = star_dir.xy * 60.0;
                vec2 star_coord2 = star_dir.xy * 120.0 + vec2(33.7, 17.3);
                vec2 star_coord3 = star_dir.xy * 200.0 + vec2(67.1, 89.2);
                
                float star1 = fract(sin(dot(star_coord1, vec2(12.9898, 78.233))) * 43758.5453);
                float star2 = fract(sin(dot(star_coord2, vec2(39.3467, 91.456))) * 22578.1459);
                float star3 = fract(sin(dot(star_coord3, vec2(67.5324, 45.123))) * 63421.8765);
                
                vec3 star_color = vec3(0.0);
                
                // Bright stars (rare)
                if (star1 > 0.998) {{
                    float brightness = (star1 - 0.998) * 400.0;
                    star_color += vec3(1.0, 0.9, 0.8) * brightness;
                }}
                
                // Medium stars
                if (star2 > 0.996) {{
                    float brightness = (star2 - 0.996) * 200.0;
                    star_color += vec3(0.8, 0.8, 1.0) * brightness;
                }}
                
                // Dim background stars
                if (star3 > 0.994) {{
                    float brightness = (star3 - 0.994) * 100.0;
                    star_color += vec3(0.6, 0.7, 1.0) * brightness;
                }}
                
                // Deep space background with subtle cosmic glow
                vec3 cosmic_bg = vec3(0.005, 0.008, 0.015);
                
                // Add subtle nebula-like gradient
                float nebula = 0.01 * (0.5 + 0.5 * sin(star_dir.x * 2.0)) * (0.5 + 0.5 * sin(star_dir.y * 1.5));
                cosmic_bg += vec3(0.02, 0.01, 0.03) * nebula;
                
                star_color += cosmic_bg;
                
                color = vec4(star_color, 1.0);
            }}
        
            imageStore(outImage, pix, color);
        }}
        """
        
        # Grid shaders (matching C++ reference)
        grid_vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 viewProj;
        void main() {
            gl_Position = viewProj * vec4(aPos, 1.0);
        }
        """
        
        grid_fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(0.3, 0.8, 1.0, 0.8); // bright cyan lines for better visibility
        }
        """
        
        # Compile shaders
        try:
            self.display_program = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
            self.compute_program = compileProgram(
                compileShader(compute_shader, GL_COMPUTE_SHADER)
            )
            self.grid_program = compileProgram(
                compileShader(grid_vertex_shader, GL_VERTEX_SHADER),
                compileShader(grid_fragment_shader, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"❌ Shader compilation error: {e}")
            sys.exit(1)
    
    def setup_fullscreen_quad(self):
        # Fullscreen quad with texture coordinates
        quad_vertices = np.array([
            # positions   # texCoords
            -1.0,  1.0,  0.0, 1.0,  # top left
            -1.0, -1.0,  0.0, 0.0,  # bottom left
             1.0, -1.0,  1.0, 0.0,  # bottom right
            -1.0,  1.0,  0.0, 1.0,  # top left
             1.0, -1.0,  1.0, 0.0,  # bottom right
             1.0,  1.0,  1.0, 1.0   # top right
        ], dtype=np.float32)
        
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
        
        # Create texture for compute shader output
        self.output_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.compute_width, self.compute_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def render(self):
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Run compute shader for ray tracing
        self.dispatch_compute()
        
        # Draw fullscreen quad with compute shader result
        glUseProgram(self.display_program)
        glBindVertexArray(self.quad_vao)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glUniform1i(glGetUniformLocation(self.display_program, "screenTexture"), 0)
        
        glDisable(GL_DEPTH_TEST)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glEnable(GL_DEPTH_TEST)
        
        # Calculate view-projection matrix for grid
        cam_pos = self.camera.position()
        target = self.camera.target()
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Create view matrix
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Construct view matrix manually
        view_matrix = np.array([
            [right[0], up[0], -forward[0], 0],
            [right[1], up[1], -forward[1], 0],
            [right[2], up[2], -forward[2], 0],
            [-np.dot(right, cam_pos), -np.dot(up, cam_pos), np.dot(forward, cam_pos), 1]
        ], dtype=np.float32)
        
        # Create projection matrix
        fov_rad = math.radians(60.0)
        aspect = float(self.width) / float(self.height)
        near = 1e9
        far = 1e14
        
        f = 1.0 / math.tan(fov_rad / 2.0)
        proj_matrix = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        # Combine matrices
        view_proj_matrix = np.dot(proj_matrix, view_matrix)
        
        # Note: Grid is now embedded in the accretion disk for better lensing visualization
        
        pygame.display.flip()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.camera.dragging = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.camera.dragging = False
                    self.camera.moving = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.camera.dragging:
                    dx, dy = event.rel
                    self.camera.azimuth += dx * self.camera.orbit_speed
                    self.camera.elevation -= dy * self.camera.orbit_speed
                    self.camera.elevation = max(0.01, min(math.pi - 0.01, self.camera.elevation))
                    self.camera.moving = True
            
            elif event.type == pygame.MOUSEWHEEL:
                # Adaptive zoom speed based on current distance
                zoom_factor = self.camera.radius / 1e11  # Scale zoom with distance
                zoom_amount = event.y * self.camera.zoom_speed * zoom_factor
                self.camera.radius -= zoom_amount
                self.camera.radius = max(self.camera.min_radius, min(self.camera.max_radius, self.camera.radius))
                self.camera.moving = True
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Zoom in with keyboard
                    zoom_factor = self.camera.radius / 1e11
                    self.camera.radius -= self.camera.zoom_speed * zoom_factor * 2
                    self.camera.radius = max(self.camera.min_radius, self.camera.radius)
                    self.camera.moving = True
                elif event.key == pygame.K_MINUS:
                    # Zoom out with keyboard
                    zoom_factor = self.camera.radius / 1e11
                    self.camera.radius += self.camera.zoom_speed * zoom_factor * 2
                    self.camera.radius = min(self.camera.max_radius, self.camera.radius)
                    self.camera.moving = True
        
        return True
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            
            # Update camera state
            self.camera.update()
            
            # Render frame
            self.render()
            
            # Gradually reset moving state
            if self.camera.moving and not self.camera.dragging:
                self.camera.moving = False
            
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    parser = argparse.ArgumentParser(description='Black Hole Simulation with Gravitational Lensing')
    parser.add_argument('--width', type=int, default=720, help='Window width (default: 720)')
    parser.add_argument('--height', type=int, default=480, help='Window height (default: 480)')
    parser.add_argument('--grid', action='store_true', help='Enable grid pattern on accretion disk')
    parser.add_argument('--resolution', type=str, help='Set resolution (e.g., 1920x1080, 1280x720)')
    
    args = parser.parse_args()
    
    # Handle resolution shorthand
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            args.width, args.height = width, height
        except ValueError:
            print("❌ Invalid resolution format. Use format like '1920x1080'")
            sys.exit(1)

    sim = BlackHoleSimulation(width=args.width, height=args.height, enable_grid=args.grid)
    sim.run()

if __name__ == "__main__":
    main()
