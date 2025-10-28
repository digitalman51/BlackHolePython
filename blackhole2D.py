import pygame
import numpy as np
import math
import sys
from typing import List, Tuple

# Physical constants
c = 299792458.0  # speed of light
G = 6.67430e-11  # gravitational constant

class BlackHole:
    """Black hole with Schwarzschild radius"""
    def __init__(self, position: Tuple[float, float, float], mass: float):
        self.position = np.array(position)
        self.mass = mass
        self.r_s = 2.0 * G * mass / (c * c)  # Schwarzschild radius

class Ray:
    """Ray following geodesic paths in Schwarzschild spacetime"""
    def __init__(self, pos: Tuple[float, float], direction: Tuple[float, float], black_hole, color=(255, 255, 255)):
        # Cartesian coordinates
        self.x = pos[0]
        self.y = pos[1]
        
        # Convert to polar coordinates
        self.r = math.sqrt(self.x*self.x + self.y*self.y)
        self.phi = math.atan2(self.y, self.x)
        
        # Initial velocities in polar coordinates
        self.dr = direction[0] * math.cos(self.phi) + direction[1] * math.sin(self.phi)
        self.dphi = (-direction[0] * math.sin(self.phi) + direction[1] * math.cos(self.phi)) / self.r
        
        # Store conserved quantities
        self.L = self.r*self.r * self.dphi
        f = 1.0 - black_hole.r_s/self.r
        dt_dl = math.sqrt((self.dr*self.dr)/(f*f) + (self.r*self.r*self.dphi*self.dphi)/f)
        self.E = f * dt_dl
        
        # Visual properties
        self.trail = [(self.x, self.y)]
        self.active = True
        self.color = color
        self.max_trail_length = 500  # Shorter for better performance
        
    def step(self, dl: float, rs: float):
        """Advance ray one step using RK4 integration"""
        if not self.active:
            return
            
        # Check if ray is captured by black hole
        if self.r <= rs:
            self.active = False
            self.trail.clear()  # Immediately clear trail when captured!
            return
            
        # RK4 integration
        self._rk4_step(dl, rs)
        
        # Convert back to Cartesian coordinates
        self.x = self.r * math.cos(self.phi)
        self.y = self.r * math.sin(self.phi)
        
        # Add to trail with length limit
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
        
        # Stop if ray goes too far
        if self.r > 1e13:
            self.active = False
    
    def _geodesic_rhs(self, r: float, phi: float, dr: float, dphi: float, rs: float):
        """Schwarzschild null geodesic equations"""
        f = 1.0 - rs/r
        dt_dl = self.E / f
        
        return np.array([
            dr,  # dr/dλ
            dphi,  # dφ/dλ
            (-(rs/(2*r*r)) * f * (dt_dl*dt_dl) + (rs/(2*r*r*f)) * (dr*dr) + (r - rs) * (dphi*dphi)),  # d²r/dλ²
            -2.0 * dr * dphi / r  # d²φ/dλ²
        ])
    
    def _rk4_step(self, dl: float, rs: float):
        """4th-order Runge-Kutta integration"""
        y0 = np.array([self.r, self.phi, self.dr, self.dphi])
        
        k1 = self._geodesic_rhs(y0[0], y0[1], y0[2], y0[3], rs)
        temp = y0 + k1 * (dl/2.0)
        k2 = self._geodesic_rhs(temp[0], temp[1], temp[2], temp[3], rs)
        
        temp = y0 + k2 * (dl/2.0)
        k3 = self._geodesic_rhs(temp[0], temp[1], temp[2], temp[3], rs)
        
        temp = y0 + k3 * dl
        k4 = self._geodesic_rhs(temp[0], temp[1], temp[2], temp[3], rs)
        
        # Update state
        self.r    += (dl/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        self.phi  += (dl/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        self.dr   += (dl/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        self.dphi += (dl/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

class BlackHoleSimulation:    
    def __init__(self, width=1200, height=900):
        pygame.init()
        
        # Screen setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Black Hole Gravitational Lensing 2D")
        
        # Simulation parameters
        self.viewport_width = 1e12  # meters
        self.viewport_height = 7.5e11  # meters
        self.zoom_level = 1.0  # Zoom factor
        self.min_zoom = 0.1   # Can zoom in 10x
        self.max_zoom = 5.0   # Can zoom out 5x
        self.black_hole = BlackHole((0.0, 0.0, 0.0), 8.54e36)
        
        # Simulation state
        self.rays = []
        self.paused = False
        self.speed_multiplier = 1
        self.step_size = 3.0
        self.ray_angle = 0.0  # radians
        
        # Mouse interaction
        self.mouse_pressed = False
        self.mouse_start = None
        
        # Colors
        self.colors = [
            (255, 255, 255),  # White
        ]
        
        # Font for text
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
        
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates with zoom"""
        effective_width = self.viewport_width * self.zoom_level
        effective_height = self.viewport_height * self.zoom_level
        screen_x = int((world_x / effective_width + 0.5) * self.width)
        screen_y = int((-world_y / effective_height + 0.5) * self.height)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates with zoom"""
        effective_width = self.viewport_width * self.zoom_level
        effective_height = self.viewport_height * self.zoom_level
        world_x = (screen_x / self.width - 0.5) * effective_width
        world_y = -(screen_y / self.height - 0.5) * effective_height
        return world_x, world_y
    
    def add_ray(self, pos: Tuple[float, float], direction: Tuple[float, float]):
        """Add a new ray to the simulation"""
        color = self.colors[len(self.rays) % len(self.colors)]
        ray = Ray(pos, direction, self.black_hole, color)
        self.rays.append(ray)
        return ray
    
    def add_parallel_rays(self):
        """Add parallel demonstration rays"""
        start_x = -4e11
        direction = (c, 0.0)
        
        for i in range(8):
            y_pos = -1.5e11 + i * 3.75e10
            self.add_ray((start_x, y_pos), direction)
        
        print("🎯 Added parallel ray demo")
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("⏸️ Paused" if self.paused else "▶️ Resumed")
                
                elif event.key == pygame.K_r:
                    self.rays.clear()
                    print("🧹 Cleared all rays")
                
                elif event.key == pygame.K_c:
                    initial_count = len(self.rays)
                    self.rays = [ray for ray in self.rays if ray.active]
                    removed_count = initial_count - len(self.rays)
                    if removed_count > 0:
                        print(f"🗑️ Removed {removed_count} captured rays")
                
                elif event.key == pygame.K_p:
                    self.add_parallel_rays()
                
                elif event.key == pygame.K_a:
                    self.ray_angle += math.radians(10)
                    if self.ray_angle > math.pi:
                        self.ray_angle -= 2 * math.pi
                
                elif event.key == pygame.K_s:
                    self.ray_angle -= math.radians(10)
                    if self.ray_angle < -math.pi:
                        self.ray_angle += 2 * math.pi
                
                elif event.key == pygame.K_x:
                    self.ray_angle = 0.0
                
                elif event.key == pygame.K_q:
                    self.speed_multiplier = min(20, self.speed_multiplier + 1)
                    self.step_size = self.speed_multiplier * 0.6
                
                elif event.key == pygame.K_w:
                    self.speed_multiplier = max(1, self.speed_multiplier - 1)
                    self.step_size = self.speed_multiplier * 0.6
                
                elif event.key == pygame.K_z:
                    # Zoom in
                    self.zoom_level = max(self.min_zoom, self.zoom_level - 0.2)
                
                elif event.key == pygame.K_v:
                    # Zoom out  
                    self.zoom_level = min(self.max_zoom, self.zoom_level + 0.2)
                
                elif event.key == pygame.K_b:
                    # Reset zoom
                    self.zoom_level = 1.0
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_pressed = True
                    self.mouse_start = pygame.mouse.get_pos()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.mouse_pressed:
                    mouse_end = pygame.mouse.get_pos()
                    self.create_ray_from_mouse(self.mouse_start, mouse_end)
                    self.mouse_pressed = False
                    self.mouse_start = None
            
            elif event.type == pygame.MOUSEWHEEL:
                # Mouse wheel zoom
                if event.y > 0:  # Scroll up - zoom in
                    self.zoom_level = max(self.min_zoom, self.zoom_level - 0.1)
                elif event.y < 0:  # Scroll down - zoom out
                    self.zoom_level = min(self.max_zoom, self.zoom_level + 0.1)
        
        return True
    
    def create_ray_from_mouse(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]):
        """Create ray from mouse interaction"""
        world_start = self.screen_to_world(*start_pos)
        
        # Calculate direction
        if abs(end_pos[0] - start_pos[0]) > 5 or abs(end_pos[1] - start_pos[1]) > 5:
            # User dragged - use mouse direction
            world_end = self.screen_to_world(*end_pos)
            dx = world_end[0] - world_start[0]
            dy = world_end[1] - world_start[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                angle = math.atan2(dy, dx)
                self.ray_angle = angle
        else:
            # Single click - use current angle
            angle = self.ray_angle
        
        # Create ray
        direction = (c * math.cos(angle), c * math.sin(angle))
        self.add_ray(world_start, direction)
    
    def update_simulation(self):
        """Update the physics simulation"""
        if self.paused:
            return
        
        # Take multiple steps for speed
        for _ in range(self.speed_multiplier):
            for ray in self.rays:
                if ray.active:
                    ray.step(self.step_size, self.black_hole.r_s)
        
        # Clean up captured rays periodically
        if len(self.rays) > 50:
            self.rays = [ray for ray in self.rays if ray.active or ray.r > self.black_hole.r_s * 2]
    
    def draw(self):
        """Draw everything to the screen"""
        # Clear screen
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw grid
        self.draw_grid()
        
        # Draw black hole
        self.draw_black_hole()
        
        # Draw photon sphere
        self.draw_photon_sphere()
        
        # Draw rays
        self.draw_rays()
        
        # Draw UI
        self.draw_ui()
        
        # Draw mouse direction line if dragging
        if self.mouse_pressed and self.mouse_start:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(self.screen, (255, 255, 0), self.mouse_start, mouse_pos, 2)
        
        pygame.display.flip()
    
    def draw_grid(self):
        """Draw reference grid"""
        grid_color = (40, 40, 40)
        spacing = self.width // 10
        
        for i in range(0, self.width + 1, spacing):
            pygame.draw.line(self.screen, grid_color, (i, 0), (i, self.height))
        for i in range(0, self.height + 1, spacing):
            pygame.draw.line(self.screen, grid_color, (0, i), (self.width, i))
    
    def draw_black_hole(self):
        """Draw the black hole"""
        center = self.world_to_screen(0, 0)
        radius_world = self.black_hole.r_s
        effective_width = self.viewport_width * self.zoom_level
        radius_screen = int(radius_world / effective_width * self.width)
        
        if radius_screen > 1:
            # Event horizon (black with red border)
            pygame.draw.circle(self.screen, (0, 0, 0), center, radius_screen)
            pygame.draw.circle(self.screen, (255, 0, 0), center, radius_screen, max(1, radius_screen // 10))
    
    def draw_photon_sphere(self):
        """Draw the photon sphere"""
        center = self.world_to_screen(0, 0)
        radius_world = 1.5 * self.black_hole.r_s
        effective_width = self.viewport_width * self.zoom_level
        radius_screen = int(radius_world / effective_width * self.width)
        
        if radius_screen > 1:
            pygame.draw.circle(self.screen, (255, 255, 0), center, radius_screen, max(1, radius_screen // 20))
    
    def draw_rays(self):
        """Draw all rays and their trails"""
        for ray in self.rays:
            if len(ray.trail) < 2:
                continue
            
            # Draw trail
            points = []
            for i, (x, y) in enumerate(ray.trail):
                screen_pos = self.world_to_screen(x, y)
                if 0 <= screen_pos[0] < self.width and 0 <= screen_pos[1] < self.height:
                    points.append(screen_pos)
            
            if len(points) > 1:
                # Draw trail with fading effect
                for i in range(len(points) - 1):
                    alpha = (i + 1) / len(points)
                    color = tuple(int(c * alpha) for c in ray.color)
                    try:
                        pygame.draw.line(self.screen, color, points[i], points[i + 1], 1)
                    except:
                        pass  # Skip invalid points
            
            # Draw current position
            if ray.active and points:
                pygame.draw.circle(self.screen, ray.color, points[-1], 3)
    
    def draw_ui(self):
        """Draw user interface text"""
        # Status info
        active_rays = sum(1 for ray in self.rays if ray.active)
        total_rays = len(self.rays)
        captured_rays = total_rays - active_rays
        
        status_text = f"Rays: {total_rays} | Active: {active_rays} | Captured: {captured_rays}"
        if self.paused:
            status_text += " | PAUSED"
        else:
            status_text += f" | Speed: {self.speed_multiplier}x"
        
        text = self.font.render(status_text, True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        # Current angle and zoom
        angle_text = f"Next Ray Angle: {math.degrees(self.ray_angle):.1f}°"
        text = self.font.render(angle_text, True, (255, 255, 0))
        self.screen.blit(text, (10, 40))
        
        zoom_text = f"Zoom: {self.zoom_level:.1f}x (Z/V or Mouse Wheel, B to reset)"
        text = self.font.render(zoom_text, True, (0, 255, 255))
        self.screen.blit(text, (10, 70))
        
        # Controls
        controls = [
            "Controls:",
            "Click & Drag: Set ray direction",
            "A/S: Adjust angle (±10°) | Q/W: Speed ±1x | Z/V/Wheel: Zoom",
            "SPACE: Pause | R: Reset | C: Clear captured | P: Parallel demo",
            "X: Reset angle | B: Reset zoom"
        ]
        
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, (0, 255, 255))
            self.screen.blit(text, (10, self.height - 100 + i * 20))
    
    def run(self):
        
        # Add initial demo
        self.add_parallel_rays()
        
        # Main loop
        running = True
        while running:
            running = self.handle_events()
            self.update_simulation()
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        sys.exit()

def main():
    sim = BlackHoleSimulation()
    sim.run()

if __name__ == "__main__":
    main()
