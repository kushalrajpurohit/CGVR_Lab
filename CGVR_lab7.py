
import numpy as np
import matplotlib.pyplot as plt

print("PHONG ILLUMINATION MODEL")
print("------------------------")

ka = float(input("Enter ambient coefficient (ka, 0-1): "))
kd = float(input("Enter diffuse coefficient (kd, 0-1): "))
ks = float(input("Enter specular coefficient (ks, 0-1): "))
shininess = float(input("Enter shininess value (e.g. 16, 32, 64): "))
ambient_light = float(input("Enter ambient light intensity (0-1): "))
light_intensity = float(input("Enter point light intensity (0-1): "))
light_x = float(input("Enter light X position (-5 to 5): "))
light_y = float(input("Enter light Y position (-5 to 5): "))
light_z = float(input("Enter light Z position (1 to 10): "))

if not (0 <= ka <= 1 and 0 <= kd <= 1 and 0 <= ks <= 1):
    raise ValueError("Lighting coefficients must be between 0 and 1.")

if shininess <= 0:
    raise ValueError("Shininess must be greater than zero.")

if not (0 <= ambient_light <= 1 and 0 <= light_intensity <= 1):
    raise ValueError("Light intensities must be between 0 and 1.")

resolution = 250
theta = np.linspace(0, 2 * np.pi, resolution)
phi = np.linspace(0, np.pi, resolution)

X = np.outer(np.cos(theta), np.sin(phi))
Y = np.outer(np.sin(theta), np.sin(phi))
Z = np.outer(np.ones_like(theta), np.cos(phi))

normals = np.stack((X, Y, Z), axis=-1)
normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

light_position = np.array([light_x, light_y, light_z])
light_vectors = light_position - normals
light_vectors /= np.maximum(
    np.linalg.norm(light_vectors, axis=-1, keepdims=True), 1e-8
)

view_position = np.array([0.0, 0.0, 5.0])
view_vectors = view_position - normals
view_vectors /= np.maximum(
    np.linalg.norm(view_vectors, axis=-1, keepdims=True), 1e-8
)

diffuse = np.maximum(np.sum(normals * light_vectors, axis=-1), 0)

reflection = 2 * diffuse[..., None] * normals - light_vectors
reflection /= np.maximum(
    np.linalg.norm(reflection, axis=-1, keepdims=True), 1e-8
)

specular = np.maximum(
    np.sum(reflection * view_vectors, axis=-1), 0
) ** shininess

specular[diffuse <= 0] = 0

intensity = (
    ka * ambient_light
    + kd * light_intensity * diffuse
    + ks * light_intensity * specular
)

intensity = np.clip(intensity, 0, 1)

base_color = np.array([0.2, 0.55, 1.0])
surface_colors = intensity[..., None] * base_color

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X, Y, Z,
    facecolors=surface_colors,
    linewidth=0,
    antialiased=True,
    shade=False
)

ax.set_title("Phong Illumination Model - 3D Sphere")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)

print("\nPhong illumination calculated successfully.")
print(f"Minimum intensity: {intensity.min():.3f}")
print(f"Maximum intensity: {intensity.max():.3f}")

plt.show()
