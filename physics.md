# Physics of Black Hole Simulation

## Introduction

This document explains the physics concepts implemented in the black hole simulation, covering both the 2D geodesic ray tracing and the 3D backward ray tracing approaches. The simulations demonstrate gravitational lensing effects around a Schwarzschild black hole using accurate general relativistic calculations.

## Fundamental Physics Concepts

### 1. The Schwarzschild Metric

The foundation of our simulation is the Schwarzschild metric, which describes the spacetime geometry around a spherically symmetric, non-rotating black hole:

$$
ds^2 = -\left(1 - \frac{r_s}{r}\right)c^2dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1}dr^2 + r^2(d\theta^2 + \sin^2\theta \, d\phi^2)
$$

Where:

- $r_s = \frac{2GM}{c^2}$ is the Schwarzschild radius (event horizon)
- $G = 6.67430 \times 10^{-11} \text{ m}^3 \text{ kg}^{-1} \text{ s}^{-2}$ is the gravitational constant
- $M$ is the black hole mass
- $c = 299,792,458 \text{ m/s}$ is the speed of light

**Implementation in code:**

```python
def __init__(self, mass: float):
    self.mass = mass
    self.r_s = 2.0 * G * mass / (c * c)  # Schwarzschild radius
```

### 2. Geodesic Equations

Massive and massless particles follow geodesics in curved spacetime. For light rays (null geodesics), the equations of motion are derived from the Lagrangian approach.

## 2D Implementation: Forward Ray Tracing

The 2D simulation traces light rays forward in time using polar coordinates (r, φ) in the equatorial plane.

### Conserved Quantities

For null geodesics in Schwarzschild spacetime, two quantities are conserved:

1. **Energy per unit mass:**

   $$
   E = \left(1 - \frac{r_s}{r}\right) \frac{dt}{d\lambda}
   $$
2. **Angular momentum per unit mass:**

   $$
   L = r^2 \frac{d\phi}{d\lambda}
   $$

Where $\lambda$ is an affine parameter along the geodesic.

**Implementation:**

```python
# Store conserved quantities
self.L = self.r*self.r * self.dphi
f = 1.0 - black_hole.r_s/self.r
dt_dl = math.sqrt((self.dr*self.dr)/(f*f) + (self.r*self.r*self.dphi*self.dphi)/f)
self.E = f * dt_dl
```

### Geodesic Equations in 2D

The second-order differential equations for null geodesics in polar coordinates are:

$$
\frac{d^2r}{d\lambda^2} = -\frac{r_s}{2r^2}f\left(\frac{dt}{d\lambda}\right)^2 + \frac{r_s}{2r^2f}\left(\frac{dr}{d\lambda}\right)^2 + (r - r_s)\left(\frac{d\phi}{d\lambda}\right)^2
$$

$$
\frac{d^2\phi}{d\lambda^2} = -\frac{2}{r}\frac{dr}{d\lambda}\frac{d\phi}{d\lambda}
$$

Where $f = 1 - \frac{r_s}{r}$.

**Implementation:**

```python
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
```

### Numerical Integration: Runge-Kutta 4th Order

The geodesic equations are integrated using the RK4 method for stability and accuracy:

```python
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
```

## 3D Implementation: Backward Ray Tracing

The 3D simulation uses **backward ray tracing**, a powerful technique where rays are traced from the camera backward into the scene. This is computationally efficient for rendering as each pixel corresponds to exactly one ray.

### Spherical Coordinates in 3D

In 3D, we use spherical coordinates $(r, \theta, \phi)$ where:

- $r$ is the radial distance
- $\theta$ is the polar angle $(0 \text{ to } \pi)$
- $\phi$ is the azimuthal angle $(0 \text{ to } 2\pi)$

### Ray Initialization

Each pixel on the screen corresponds to a ray direction. The ray is initialized in Cartesian coordinates and converted to spherical:

**GLSL Compute Shader Implementation:**

```glsl
Ray initRay(vec3 pos, vec3 dir) {
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
}
```

### 3D Geodesic Equations

The complete geodesic equations in spherical coordinates for Schwarzschild spacetime:

$$
\frac{d^2r}{d\lambda^2} = -\frac{r_s}{2r^2}f\left(\frac{dt}{d\lambda}\right)^2 + \frac{r_s}{2r^2f}\left(\frac{dr}{d\lambda}\right)^2 + r\left(\frac{d\theta}{d\lambda}\right)^2 + r\sin^2\theta\left(\frac{d\phi}{d\lambda}\right)^2
$$

$$
\frac{d^2\theta}{d\lambda^2} = -\frac{2}{r}\frac{dr}{d\lambda}\frac{d\theta}{d\lambda} + \sin\theta\cos\theta\left(\frac{d\phi}{d\lambda}\right)^2
$$

$$
\frac{d^2\phi}{d\lambda^2} = -\frac{2}{r}\frac{dr}{d\lambda}\frac{d\phi}{d\lambda} - \frac{2\cos\theta}{\sin\theta}\frac{d\theta}{d\lambda}\frac{d\phi}{d\lambda}
$$

**GLSL Implementation:**

```glsl
void geodesicRHS(Ray ray, out vec3 d1, out vec3 d2) {
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
}
```

### Backward Ray Tracing Process

1. **Ray Generation:** For each pixel, generate a ray from the camera through that pixel
2. **Ray Integration:** Integrate the geodesic equations backward in time
3. **Intersection Testing:** Check for intersections with:
   - Black hole event horizon (ray captured)
   - Accretion disk (in equatorial plane)
   - Other objects in the scene
4. **Color Determination:** Based on what the ray hits, determine the pixel color

### Advantages of Backward Ray Tracing

- **Efficiency:** Only rays that contribute to the final image are computed
- **Accuracy:** Each pixel gets exactly one ray, avoiding aliasing issues
- **Flexibility:** Easy to implement complex lighting models and multiple objects
- **GPU Friendly:** Highly parallelizable on modern graphics hardware

## Physical Phenomena Demonstrated

### 1. Gravitational Lensing

Light rays bend when passing near massive objects due to spacetime curvature. The simulation shows:

- **Light bending:** Straight lines in flat space become curved geodesics
- **Einstein rings:** When source, lens, and observer are aligned
- **Multiple images:** Single objects can appear multiple times

### 2. Event Horizon

The Schwarzschild radius defines the point of no return:

$$
r_s = \frac{2GM}{c^2}
$$

For Sagittarius A* ($M \approx 8.54 \times 10^{36} \text{ kg}$):

$$
r_s \approx 2.53 \times 10^{10} \text{ m} \approx 169 \text{ AU}
$$

### 3. Photon Sphere

Located at $r = 1.5r_s$, this is where photons can orbit the black hole:

**Implementation:**

```python
def draw_photon_sphere(self):
    radius_world = 1.5 * self.black_hole.r_s
    # Draw as yellow circle
```

### 4. Accretion Disk Physics

The simulation includes a realistic accretion disk with:

- **Temperature gradients:** Hotter near the black hole, cooler at edges
- **Gravitational redshift:** Light frequency shifts due to gravitational field
- **Doppler effects:** Due to orbital motion of disk material

**Color Temperature Implementation:**

```glsl
// Realistic accretion disk colors
vec3 hot_core = vec3(1.2, 1.0, 0.9);      // White-hot center
vec3 medium_temp = vec3(1.0, 0.7, 0.3);   // Orange
vec3 cooler_edge = vec3(0.9, 0.4, 0.15);  // Red-orange
vec3 cold_outer = vec3(0.6, 0.2, 0.08);   // Deep red-brown
```

## Numerical Considerations

### Step Size Selection

The integration step size $d\lambda$ must be chosen carefully:

- **2D simulation:** $\text{step\_size} = 3.0$ (adjustable with speed multiplier)
- **3D simulation:** $D\_LAMBDA = 1 \times 10^7$ (high quality), $5 \times 10^6$ (video recording)

### Coordinate Singularities

Special handling is required at:

- $r = 0$ (coordinate singularity)
- $\theta = 0, \pi$ (coordinate singularity)
- $r = r_s$ (event horizon)

### Stability and Accuracy

The RK4 integration method provides:

- 4th order accuracy: $O(h^5)$ truncation error
- Good stability properties for orbital motion
- Conservation of energy and angular momentum to machine precision
