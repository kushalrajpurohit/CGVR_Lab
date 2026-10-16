
import numpy as np
import matplotlib.pyplot as plt

print("3D VIEWING PIPELINE")
print("-------------------")

focal_length = float(input("Enter focal length (e.g. 5): "))
viewport_width = int(input("Enter viewport width (e.g. 800): "))
viewport_height = int(input("Enter viewport height (e.g. 600): "))

if focal_length <= 0 or viewport_width <= 0 or viewport_height <= 0:
    raise ValueError("All dimensions and focal length must be positive.")

num_points = int(input("Enter number of 3D points: "))

if num_points < 1:
    raise ValueError("Enter at least one point.")

points_3d = []

for i in range(num_points):
    print(f"\nPoint {i + 1}:")
    x = float(input("Enter X: "))
    y = float(input("Enter Y: "))
    z = float(input("Enter Z (must be positive): "))

    if z <= 0:
        raise ValueError("Z must be positive for this projection.")

    points_3d.append([x, y, z])

points_3d = np.array(points_3d)

projected_points = []

for x, y, z in points_3d:
    xp = focal_length * x / z
    yp = focal_length * y / z
    projected_points.append([xp, yp])

projected_points = np.array(projected_points)

viewport_points = []

for xp, yp in projected_points:
    u = (xp + 1) * viewport_width / 2
    v = (1 - yp) * viewport_height / 2
    viewport_points.append([u, v])

viewport_points = np.array(viewport_points)

print("\n3D Viewing Pipeline Results")
print("---------------------------")
print("Point       3D Coordinates       Projected (x, y)       Viewport (u, v)")

for i in range(num_points):
    x, y, z = points_3d[i]
    xp, yp = projected_points[i]
    u, v = viewport_points[i]

    print(
        f"{i + 1:<12}"
        f"({x:.2f}, {y:.2f}, {z:.2f})     "
        f"({xp:.2f}, {yp:.2f})             "
        f"({u:.2f}, {v:.2f})"
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    points_3d[:, 0],
    points_3d[:, 1],
    s=60
)

for i, (x, y, z) in enumerate(points_3d):
    axes[0].text(x, y, f"P{i + 1}")

axes[0].set_title("3D Points: X-Y View")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].grid(True)
axes[0].axis("equal")

axes[1].scatter(
    viewport_points[:, 0],
    viewport_points[:, 1],
    s=60
)

for i, (u, v) in enumerate(viewport_points):
    axes[1].text(u, v, f"P{i + 1}")

axes[1].set_title("2D Viewport Mapping")
axes[1].set_xlabel("Viewport X (pixels)")
axes[1].set_ylabel("Viewport Y (pixels)")
axes[1].set_xlim(0, viewport_width)
axes[1].set_ylim(viewport_height, 0)
axes[1].grid(True)

plt.tight_layout()
plt.show()
