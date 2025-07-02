import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import math
import time

# =================================================================
# COMPUTER GRAPHICS ALGORITHMS IMPLEMENTATION
# =================================================================

# ---------------------
# Region codes for clipping
# ---------------------
INSIDE = 0  # 0000
LEFT   = 1  # 0001
RIGHT  = 2  # 0010
BOTTOM = 4  # 0100
TOP    = 8  # 1000

# =================================================================
# RASTERIZATION ALGORITHMS
# =================================================================

def dda_line(x0, y0, x1, y1):
    """Digital Differential Analyzer line drawing algorithm"""
    points = []
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    
    if steps == 0:
        return [(x0, y0)]
    
    x_inc = dx / steps
    y_inc = dy / steps
    x = x0
    y = y0
    
    for _ in range(int(steps) + 1):
        points.append((round(x), round(y)))
        x += x_inc
        y += y_inc
        
    return points

def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line drawing algorithm"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
    return points

def midpoint_circle(radius, center=(0, 0)):
    """Midpoint circle drawing algorithm"""
    cx, cy = center
    x = radius
    y = 0
    decision = 1 - radius
    points = []
    
    while x >= y:
        points.extend([
            (cx + x, cy + y), (cx - x, cy + y),
            (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x),
            (cx + y, cy - x), (cx - y, cy - x)
        ])
        y += 1
        if decision <= 0:
            decision += 2 * y + 1
        else:
            x -= 1
            decision += 2 * (y - x) + 1
            
    return points

def bresenham_circle(radius, center=(0, 0)):
    """Bresenham's circle drawing algorithm"""
    cx, cy = center
    x = 0
    y = radius
    d = 3 - 2 * radius
    points = []
    
    while x <= y:
        points.extend([
            (cx + x, cy + y), (cx - x, cy + y),
            (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x),
            (cx + y, cy - x), (cx - y, cy - x)
        ])
        x += 1
        if d < 0:
            d += 4 * x + 6
        else:
            y -= 1
            d += 4 * (x - y) + 10
            
    return points

def midpoint_ellipse(rx, ry, center=(0, 0)):
    """Midpoint ellipse drawing algorithm"""
    cx, cy = center
    points = []
    x = 0
    y = ry
    d1 = (ry * ry) - (rx * rx * ry) + (0.25 * rx * rx)
    dx = 2 * ry * ry * x
    dy = 2 * rx * rx * y
    
    # Region 1
    while dx < dy:
        points.extend([
            (cx + x, cy + y), (cx - x, cy + y),
            (cx + x, cy - y), (cx - x, cy - y)
        ])
        x += 1
        dx += 2 * ry * ry
        if d1 < 0:
            d1 += dx + ry * ry
        else:
            y -= 1
            dy -= 2 * rx * rx
            d1 += dx - dy + ry * ry
    
    # Region 2
    d2 = (ry * ry) * (x + 0.5) * (x + 0.5) + \
         (rx * rx) * (y - 1) * (y - 1) - \
         (rx * rx * ry * ry)
    
    while y >= 0:
        points.extend([
            (cx + x, cy + y), (cx - x, cy + y),
            (cx + x, cy - y), (cx - x, cy - y)
        ])
        y -= 1
        dy -= 2 * rx * rx
        if d2 > 0:
            d2 += rx * rx - dy
        else:
            x += 1
            dx += 2 * ry * ry
            d2 += dx - dy + rx * rx
            
    return points

# =================================================================
# AREA FILLING ALGORITHMS
# =================================================================

def boundary_fill(x, y, fill_color, boundary_color, img):
    """Boundary fill algorithm (recursive)"""
    if (x < 0 or x >= img.shape[1] or 
        y < 0 or y >= img.shape[0] or 
        np.array_equal(img[y, x], boundary_color) or 
        np.array_equal(img[y, x], fill_color)):
        return
    
    img[y, x] = fill_color
    boundary_fill(x + 1, y, fill_color, boundary_color, img)
    boundary_fill(x - 1, y, fill_color, boundary_color, img)
    boundary_fill(x, y + 1, fill_color, boundary_color, img)
    boundary_fill(x, y - 1, fill_color, boundary_color, img)

def flood_fill(x, y, fill_color, target_color, img):
    """Flood fill algorithm (recursive)"""
    if (x < 0 or x >= img.shape[1] or 
        y < 0 or y >= img.shape[0] or 
        not np.array_equal(img[y, x], target_color) or 
        np.array_equal(img[y, x], fill_color)):
        return
    
    img[y, x] = fill_color
    flood_fill(x + 1, y, fill_color, target_color, img)
    flood_fill(x - 1, y, fill_color, target_color, img)
    flood_fill(x, y + 1, fill_color, target_color, img)
    flood_fill(x, y - 1, fill_color, target_color, img)

def scanline_fill(polygon):
    """Scanline polygon filling algorithm"""
    if not polygon:
        return []
    
    # Find min and max y coordinates
    y_min = int(min(y for x, y in polygon))
    y_max = int(max(y for x, y in polygon))
    
    # Create edge table
    edges = []
    n = len(polygon)
    
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        
        if y1 == y2:
            continue
            
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        inv_slope = (x2 - x1) / (y2 - y1)
        edges.append((y1, y2, x1, inv_slope))
    
    # Process scanlines
    filled_points = []
    
    for y in range(y_min, y_max + 1):
        intersections = []
        
        for edge in edges:
            y1, y2, x1, inv_slope = edge
            if y1 <= y < y2:
                x_intersect = x1 + inv_slope * (y - y1)
                intersections.append(x_intersect)
        
        intersections.sort()
        
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = int(np.ceil(intersections[i]))
                x_end = int(np.floor(intersections[i + 1]))
                for x in range(x_start, x_end + 1):
                    filled_points.append((x, y))
    
    return filled_points

# =================================================================
# 2D TRANSFORMATIONS
# =================================================================

def translate(points, tx, ty):
    """2D translation"""
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ T.T)[:, :2]

def scale(points, sx, sy, fixed=(0, 0)):
    """2D scaling relative to fixed point"""
    px, py = fixed
    T1 = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1]])
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    T2 = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ T1.T @ S.T @ T2.T)[:, :2]

def rotate(points, angle, pivot=(0, 0)):
    """2D rotation around pivot point"""
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    px, py = pivot
    T1 = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1]])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T2 = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ T1.T @ R.T @ T2.T)[:, :2]

def shear(points, shx, shy):
    """2D shearing transformation"""
    Sh = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ Sh.T)[:, :2]

def reflect(points, axis='x'):
    """2D reflection"""
    if axis == 'x':
        M = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis == 'y':
        M = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:  # origin
        M = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ M.T)[:, :2]

# =================================================================
# LINE CLIPPING ALGORITHMS
# =================================================================

def compute_outcode(x, y, xmin, ymin, xmax, ymax):
    """Compute region code for point"""
    code = INSIDE
    if x < xmin:   code |= LEFT
    elif x > xmax: code |= RIGHT
    if y < ymin:   code |= BOTTOM
    elif y > ymax: code |= TOP
    return code

def cohen_sutherland_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Cohen-Sutherland line clipping algorithm"""
    outcode0 = compute_outcode(x0, y0, xmin, ymin, xmax, ymax)
    outcode1 = compute_outcode(x1, y1, xmin, ymin, xmax, ymax)
    accept = False

    while True:
        if not (outcode0 | outcode1):  # Trivial accept
            accept = True
            break
        elif outcode0 & outcode1:      # Trivial reject
            break
        else:
            outcode_out = outcode0 if outcode0 else outcode1
            
            if outcode_out & TOP:
                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                y = ymax
            elif outcode_out & BOTTOM:
                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                y = ymin
            elif outcode_out & RIGHT:
                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                x = xmax
            elif outcode_out & LEFT:
                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                x = xmin

            if outcode_out == outcode0:
                x0, y0 = x, y
                outcode0 = compute_outcode(x0, y0, xmin, ymin, xmax, ymax)
            else:
                x1, y1 = x, y
                outcode1 = compute_outcode(x1, y1, xmin, ymin, xmax, ymax)

    if accept:
        return (x0, y0), (x1, y1)
    return None

def liang_barsky_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Liang-Barsky line clipping algorithm"""
    dx = x1 - x0
    dy = y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
    u1, u2 = 0.0, 1.0

    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return None  # Line parallel and outside
        else:
            r = q[i] / p[i]
            if p[i] < 0:
                if r > u1: u1 = r
            else:
                if r < u2: u2 = r

    if u1 > u2:
        return None

    nx0 = x0 + u1 * dx
    ny0 = y0 + u1 * dy
    nx1 = x0 + u2 * dx
    ny1 = y0 + u2 * dy
    return (nx0, ny0), (nx1, ny1)

# =================================================================
# POLYGON CLIPPING ALGORITHMS
# =================================================================

def sutherland_hodgman_clip(subject, clip_window):
    """Sutherland-Hodgman polygon clipping algorithm"""
    xmin, ymin, xmax, ymax = clip_window
    
    def clip_edge(poly, inside_fn, intersect_fn):
        output = []
        if not poly: return output
        prev = poly[-1]
        prev_inside = inside_fn(prev)
        for curr in poly:
            curr_inside = inside_fn(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect_fn(prev, curr))
            prev, prev_inside = curr, curr_inside
        return output

    def inside_left(p): return p[0] >= xmin
    def intersect_left(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        t = (xmin - x1) / (x2 - x1)
        return (xmin, y1 + t*(y2 - y1))

    def inside_right(p): return p[0] <= xmax
    def intersect_right(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        t = (xmax - x1) / (x2 - x1)
        return (xmax, y1 + t*(y2 - y1))

    def inside_bottom(p): return p[1] >= ymin
    def intersect_bottom(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        t = (ymin - y1) / (y2 - y1)
        return (x1 + t*(x2 - x1), ymin)

    def inside_top(p): return p[1] <= ymax
    def intersect_top(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        t = (ymax - y1) / (y2 - y1)
        return (x1 + t*(x2 - x1), ymax)

    clipped = subject
    for inside_fn, intersect_fn in [
        (inside_left, intersect_left),
        (inside_right, intersect_right),
        (inside_bottom, intersect_bottom),
        (inside_top, intersect_top)
    ]:
        clipped = clip_edge(clipped, inside_fn, intersect_fn)
    
    return clipped

# =================================================================
# CURVE ALGORITHMS
# =================================================================

def bezier_curve(control_points, num_points=100):
    """Bezier curve algorithm"""
    n = len(control_points) - 1
    curve = []
    
    for i in range(num_points + 1):
        t = i / num_points
        x, y = 0, 0
        for k in range(n + 1):
            # Binomial coefficient
            binom = math.comb(n, k)
            term = binom * (1 - t) ** (n - k) * t ** k
            x += term * control_points[k][0]
            y += term * control_points[k][1]
        curve.append((x, y))
    return curve

def bspline_curve(control_points, degree=3, num_points=100):
    """B-Spline curve algorithm"""
    n = len(control_points) - 1
    m = n + degree + 1
    knots = list(range(m + 1))
    
    def basis(i, d, t):
        if d == 0:
            return 1 if knots[i] <= t < knots[i+1] else 0
        denom1 = knots[i+d] - knots[i]
        denom2 = knots[i+d+1] - knots[i+1]
        term1 = 0 if denom1 == 0 else (t - knots[i]) / denom1 * basis(i, d-1, t)
        term2 = 0 if denom2 == 0 else (knots[i+d+1] - t) / denom2 * basis(i+1, d-1, t)
        return term1 + term2
    
    curve = []
    for seg in range(n - degree + 1):
        for j in range(num_points):
            t = knots[degree] + (knots[seg+degree+1] - knots[degree]) * j / num_points
            x, y = 0, 0
            for i in range(seg, seg + degree + 1):
                b = basis(i, degree, t)
                x += b * control_points[i][0]
                y += b * control_points[i][1]
            curve.append((x, y))
    return curve

# =================================================================
# 3D TRANSFORMATIONS
# =================================================================

def translate_3d(points, tx, ty, tz):
    """3D translation"""
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ T.T)[:, :3]

def scale_3d(points, sx, sy, sz, pivot=(0, 0, 0)):
    """3D scaling relative to pivot point"""
    px, py, pz = pivot
    T1 = np.array([[1,0,0,-px],[0,1,0,-py],[0,0,1,-pz],[0,0,0,1]])
    S = np.array([[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]])
    T2 = np.array([[1,0,0,px],[0,1,0,py],[0,0,1,pz],[0,0,0,1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ T1.T @ S.T @ T2.T)[:, :3]

def rotate_x_3d(points, angle_deg, pivot=(0, 0, 0)):
    """3D rotation around X-axis"""
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    px, py, pz = pivot
    T1 = np.array([[1,0,0,-px],[0,1,0,-py],[0,0,1,-pz],[0,0,0,1]])
    R = np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
    T2 = np.array([[1,0,0,px],[0,1,0,py],[0,0,1,pz],[0,0,0,1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ T1.T @ R.T @ T2.T)[:, :3]

def rotate_y_3d(points, angle_deg, pivot=(0, 0, 0)):
    """3D rotation around Y-axis"""
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    px, py, pz = pivot
    T1 = np.array([[1,0,0,-px],[0,1,0,-py],[0,0,1,-pz],[0,0,0,1]])
    R = np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
    T2 = np.array([[1,0,0,px],[0,1,0,py],[0,0,1,pz],[0,0,0,1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ T1.T @ R.T @ T2.T)[:, :3]

def rotate_z_3d(points, angle_deg, pivot=(0, 0, 0)):
    """3D rotation around Z-axis"""
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    px, py, pz = pivot
    T1 = np.array([[1,0,0,-px],[0,1,0,-py],[0,0,1,-pz],[0,0,0,1]])
    R = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
    T2 = np.array([[1,0,0,px],[0,1,0,py],[0,0,1,pz],[0,0,0,1]])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homog @ T1.T @ R.T @ T2.T)[:, :3]

def perspective_projection(points, fov=60, aspect=1, near=0.1, far=100):
    """Perspective projection transformation"""
    f = 1 / np.tan(np.radians(fov) / 2)
    proj_matrix = np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    projected = homog @ proj_matrix.T
    projected[:, :3] /= projected[:, 3:]  # Perspective divide
    return projected[:, :2]  # Return only x,y for 2D display

# =================================================================
# VISUALIZATION FUNCTIONS
# =================================================================

def visualize_line(points, title):
    """Visualize line drawing results"""
    if not points:
        print("No points to visualize")
        return
    
    plt.figure(figsize=(8, 8))
    x_vals, y_vals = zip(*points)
    plt.plot(x_vals, y_vals, 'bo-', linewidth=2, markersize=5)
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def visualize_circle(points, title):
    """Visualize circle drawing results"""
    if not points:
        print("No points to visualize")
        return
    
    plt.figure(figsize=(8, 8))
    x_vals, y_vals = zip(*points)
    plt.scatter(x_vals, y_vals, s=5)
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def visualize_transformation(original, transformed, title):
    """Visualize 2D transformations"""
    plt.figure(figsize=(8, 8))
    x_orig, y_orig = zip(*original)
    plt.plot(x_orig, y_orig, 'bo-', label='Original')
    x_trans, y_trans = zip(*transformed)
    plt.plot(x_trans, y_trans, 'ro-', label='Transformed')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def visualize_clipping(p0, p1, c0, c1, window):
    """Visualize line clipping results"""
    xmin, ymin, xmax, ymax = window
    plt.figure(figsize=(8, 8))
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b--', label='Original')
    
    if c0 and c1:
        plt.plot([c0[0], c1[0]], [c0[1], c1[1]], 'r-', linewidth=2, label='Clipped')
        plt.scatter([c0[0], c1[0]], [c0[1], c1[1]], c='red', s=50)
    
    rect_x = [xmin, xmax, xmax, xmin, xmin]
    rect_y = [ymin, ymin, ymax, ymax, ymin]
    plt.plot(rect_x, rect_y, 'g-', label='Clip Window')
    
    plt.title('Line Clipping')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def visualize_curve(control_points, curve_points, title):
    """Visualize curve algorithms"""
    plt.figure(figsize=(8, 8))
    cx, cy = zip(*control_points)
    plt.plot(cx, cy, 'bo-', label='Control Points')
    
    if curve_points:
        bx, by = zip(*curve_points)
        plt.plot(bx, by, 'r-', label='Curve')
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def visualize_3d(points, edges, title):
    """Visualize 3D transformations"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot edges
    for edge in edges:
        i, j = edge
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            [points[i, 2], points[j, 2]],
            'b-', linewidth=2
        )
    
    # Plot vertices
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], 
        c='r', marker='o', s=50, depthshade=False
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def visualize_fill(polygon, filled_points, title):
    """Visualize filling algorithms"""
    plt.figure(figsize=(8, 8))
    
    # Draw polygon
    xs, ys = zip(*(polygon + [polygon[0]]))
    plt.plot(xs, ys, 'b-', label='Polygon')
    
    # Draw filled points
    if filled_points:
        x_vals, y_vals = zip(*filled_points)
        plt.scatter(x_vals, y_vals, c='red', s=5, alpha=0.6, label='Filled Points')
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# =================================================================
# ALGORITHM CATALOG
# =================================================================

def list_all_algorithms():
    """Display all major computer graphics algorithms"""
    print("\n" + "="*60)
    print("COMPUTER GRAPHICS ALGORITHMS CATALOG")
    print("="*60)
    
    categories = {
        "Rasterization Algorithms": [
            "DDA Line Algorithm",
            "Bresenham's Line Algorithm",
            "Midpoint Line Algorithm",
            "Midpoint Circle Algorithm",
            "Bresenham's Circle Algorithm",
            "Midpoint Ellipse Algorithm"
        ],
        "Area Filling Algorithms": [
            "Boundary Fill Algorithm",
            "Flood Fill Algorithm",
            "Scanline Fill Algorithm"
        ],
        "2D Transformations": [
            "Translation",
            "Scaling",
            "Rotation",
            "Shearing",
            "Reflection"
        ],
        "Line Clipping Algorithms": [
            "Cohen-Sutherland Algorithm",
            "Liang-Barsky Algorithm"
        ],
        "Polygon Clipping Algorithms": [
            "Sutherland-Hodgman Algorithm"
        ],
        "Curve Algorithms": [
            "Bezier Curves",
            "B-Spline Curves"
        ],
        "3D Transformations": [
            "3D Translation",
            "3D Scaling",
            "3D Rotation (X, Y, Z axes)",
            "Perspective Projection"
        ],
        "Advanced Algorithms": [
            "Z-Buffer Algorithm",
            "Phong Shading",
            "Ray Tracing",
            "Texture Mapping",
            "Fractal Generation"
        ]
    }
    
    # Print all algorithms organized by category
    for category, algorithms in categories.items():
        print(f"\n{category.upper()}:")
        for i, algo in enumerate(algorithms, 1):
            print(f"  {i}. {algo}")
    
    print("\n" + "="*60)
    print(f"Total Categories: {len(categories)}")
    print(f"Total Algorithms: {sum(len(algo) for algo in categories.values())}")
    print("="*60)

# =================================================================
# MAIN MENU SYSTEM
# =================================================================

def main():
    while True:
        print("\n" + "="*60)
        print("COMPUTER GRAPHICS ALGORITHMS DEMONSTRATION")
        print("="*60)
        print("1. Rasterization Algorithms")
        print("2. Area Filling Algorithms")
        print("3. 2D Transformations")
        print("4. Line Clipping Algorithms")
        print("5. Polygon Clipping Algorithms")
        print("6. Curve Algorithms")
        print("7. 3D Transformations")
        print("8. View All Algorithms")
        print("9. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-9): ")
        
        if choice == '1':
            rasterization_menu()
        elif choice == '2':
            filling_menu()
        elif choice == '3':
            transformations_2d_menu()
        elif choice == '4':
            clipping_menu()
        elif choice == '5':
            polygon_clipping_menu()
        elif choice == '6':
            curve_menu()
        elif choice == '7':
            transformations_3d_menu()
        elif choice == '8':
            list_all_algorithms()
        elif choice == '9':
            print("Exiting program...")
            break
        else:
            print("Invalid choice! Please enter a number between 1-9.")

# =================================================================
# SUB-MENUS
# =================================================================

def rasterization_menu():
    while True:
        print("\nRASTERIZATION ALGORITHMS")
        print("1. DDA Line Algorithm")
        print("2. Bresenham's Line Algorithm")
        print("3. Midpoint Circle Algorithm")
        print("4. Bresenham's Circle Algorithm")
        print("5. Midpoint Ellipse Algorithm")
        print("6. Back to Main Menu")
        
        choice = input("Enter choice (1-6): ")
        
        if choice == '1':
            x0, y0 = map(int, input("Enter start point (x0 y0): ").split())
            x1, y1 = map(int, input("Enter end point (x1 y1): ").split())
            points = dda_line(x0, y0, x1, y1)
            visualize_line(points, "DDA Line Algorithm")
            
        elif choice == '2':
            x0, y0 = map(int, input("Enter start point (x0 y0): ").split())
            x1, y1 = map(int, input("Enter end point (x1 y1): ").split())
            points = bresenham_line(x0, y0, x1, y1)
            visualize_line(points, "Bresenham's Line Algorithm")
            
        elif choice == '3':
            r = int(input("Enter radius: "))
            cx, cy = map(int, input("Enter center (cx cy): ").split())
            points = midpoint_circle(r, (cx, cy))
            visualize_circle(points, "Midpoint Circle Algorithm")
            
        elif choice == '4':
            r = int(input("Enter radius: "))
            cx, cy = map(int, input("Enter center (cx cy): ").split())
            points = bresenham_circle(r, (cx, cy))
            visualize_circle(points, "Bresenham's Circle Algorithm")
            
        elif choice == '5':
            rx = int(input("Enter x-radius: "))
            ry = int(input("Enter y-radius: "))
            cx, cy = map(int, input("Enter center (cx cy): ").split())
            points = midpoint_ellipse(rx, ry, (cx, cy))
            visualize_circle(points, "Midpoint Ellipse Algorithm")
            
        elif choice == '6':
            break
            
        else:
            print("Invalid choice!")

def filling_menu():
    while True:
        print("\nAREA FILLING ALGORITHMS")
        print("1. Boundary Fill Algorithm")
        print("2. Flood Fill Algorithm")
        print("3. Scanline Fill Algorithm")
        print("4. Back to Main Menu")
        
        choice = input("Enter choice (1-4): ")
        
        if choice == '1':
            print("Note: This visualization uses a simple grid representation")
            size = 20
            img = np.zeros((size, size, 3), dtype=np.uint8) + 255  # White background
            
            # Draw a rectangle
            for i in range(5, 15):
                for j in range(5, 15):
                    if i == 5 or i == 14 or j == 5 or j == 14:
                        img[i, j] = [0, 0, 0]  # Black boundary
            
            start_x, start_y = 10, 10
            boundary_fill(start_x, start_y, [255, 0, 0], [0, 0, 0], img)
            
            plt.imshow(img)
            plt.title("Boundary Fill Algorithm")
            plt.show()
            
        elif choice == '2':
            print("Note: This visualization uses a simple grid representation")
            size = 20
            img = np.zeros((size, size, 3), dtype=np.uint8) + 255  # White background
            
            # Draw a circle
            for i in range(size):
                for j in range(size):
                    if (i-10)**2 + (j-10)**2 <= 25:  # Radius 5 circle
                        img[i, j] = [0, 0, 0]  # Black boundary
            
            start_x, start_y = 10, 10
            flood_fill(start_x, start_y, [0, 255, 0], [255, 255, 255], img)
            
            plt.imshow(img)
            plt.title("Flood Fill Algorithm")
            plt.show()
            
        elif choice == '3':
            n = int(input("How many vertices in the polygon? "))
            polygon = []
            for i in range(n):
                x = float(input(f"Vertex {i+1} x: "))
                y = float(input(f"Vertex {i+1} y: "))
                polygon.append((x, y))
                
            filled_points = scanline_fill(polygon)
            visualize_fill(polygon, filled_points, "Scanline Fill Algorithm")
            
        elif choice == '4':
            break
            
        else:
            print("Invalid choice!")

def transformations_2d_menu():
    # Default triangle
    points = np.array([[0, 0], [5, 10], [10, 0]])
    
    while True:
        print("\n2D TRANSFORMATIONS")
        print("1. Translation")
        print("2. Scaling")
        print("3. Rotation")
        print("4. Shearing")
        print("5. Reflection")
        print("6. Back to Main Menu")
        
        choice = input("Enter choice (1-6): ")
        
        if choice == '1':
            tx = float(input("Enter tx: "))
            ty = float(input("Enter ty: "))
            transformed = translate(points, tx, ty)
            visualize_transformation(points, transformed, "2D Translation")
            
        elif choice == '2':
            sx = float(input("Enter sx: "))
            sy = float(input("Enter sy: "))
            px = float(input("Pivot x (default 0): ") or 0)
            py = float(input("Pivot y (default 0): ") or 0)
            transformed = scale(points, sx, sy, (px, py))
            visualize_transformation(points, transformed, "2D Scaling")
            
        elif choice == '3':
            angle = float(input("Enter angle (degrees): "))
            px = float(input("Pivot x (default 0): ") or 0)
            py = float(input("Pivot y (default 0): ") or 0)
            transformed = rotate(points, angle, (px, py))
            visualize_transformation(points, transformed, "2D Rotation")
            
        elif choice == '4':
            shx = float(input("Enter shear x: "))
            shy = float(input("Enter shear y: "))
            transformed = shear(points, shx, shy)
            visualize_transformation(points, transformed, "2D Shearing")
            
        elif choice == '5':
            axis = input("Reflection axis (x/y/origin): ").lower()
            transformed = reflect(points, axis)
            visualize_transformation(points, transformed, f"2D Reflection ({axis} axis)")
            
        elif choice == '6':
            break
            
        else:
            print("Invalid choice!")

def clipping_menu():
    while True:
        print("\nLINE CLIPPING ALGORITHMS")
        print("1. Cohen-Sutherland Algorithm")
        print("2. Liang-Barsky Algorithm")
        print("3. Back to Main Menu")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == '1' or choice == '2':
            xmin, ymin = map(float, input("Enter clip window min (xmin ymin): ").split())
            xmax, ymax = map(float, input("Enter clip window max (xmax ymax): ").split())
            x0, y0 = map(float, input("Enter line start (x0 y0): ").split())
            x1, y1 = map(float, input("Enter line end (x1 y1): ").split())
            
            if choice == '1':
                result = cohen_sutherland_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax)
            else:
                result = liang_barsky_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax)
            
            if result:
                (cx0, cy0), (cx1, cy1) = result
                print(f"Clipped line: ({cx0:.2f}, {cy0:.2f}) to ({cx1:.2f}, {cy1:.2f})")
                visualize_clipping((x0, y0), (x1, y1), (cx0, cy0), (cx1, cy1), (xmin, ymin, xmax, ymax))
            else:
                print("Line completely outside clipping window")
                visualize_clipping((x0, y0), (x1, y1), None, None, (xmin, ymin, xmax, ymax))
                
        elif choice == '3':
            break
            
        else:
            print("Invalid choice!")

def polygon_clipping_menu():
    print("\nSUTHERLAND-HODGMAN POLYGON CLIPPING")
    xmin, ymin = map(float, input("Enter clip window min (xmin ymin): ").split())
    xmax, ymax = map(float, input("Enter clip window max (xmax ymax): ").split())
    
    n = int(input("How many vertices in the subject polygon? "))
    subject = []
    for i in range(n):
        x = float(input(f"Vertex {i+1} x: "))
        y = float(input(f"Vertex {i+1} y: "))
        subject.append((x, y))
        
    clipped = sutherland_hodgman_clip(subject, (xmin, ymin, xmax, ymax))
    
    if clipped:
        print("\nClipped polygon vertices:")
        for i, (x, y) in enumerate(clipped):
            print(f"Vertex {i+1}: ({x:.2f}, {y:.2f})")
        
        plt.figure(figsize=(8, 8))
        # Original polygon
        xs, ys = zip(*(subject + [subject[0]]))
        plt.plot(xs, ys, 'b--', label='Original')
        # Clipped polygon
        xs, ys = zip(*(clipped + [clipped[0]]))
        plt.plot(xs, ys, 'r-', linewidth=2, label='Clipped')
        # Clipping window
        rect_x = [xmin, xmax, xmax, xmin, xmin]
        rect_y = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(rect_x, rect_y, 'g-', label='Clip Window')
        
        plt.title('Sutherland-Hodgman Polygon Clipping')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    else:
        print("Polygon completely outside clipping window")
        
    input("Press Enter to continue...")

def curve_menu():
    while True:
        print("\nCURVE ALGORITHMS")
        print("1. Bezier Curve")
        print("2. B-Spline Curve")
        print("3. Back to Main Menu")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == '1' or choice == '2':
            n = int(input("Number of control points: "))
            control_points = []
            for i in range(n):
                x = float(input(f"Control point {i+1} x: "))
                y = float(input(f"Control point {i+1} y: "))
                control_points.append((x, y))
            
            if choice == '1':
                curve_points = bezier_curve(control_points)
                visualize_curve(control_points, curve_points, "Bezier Curve")
            else:
                degree = min(3, n-1)
                curve_points = bspline_curve(control_points, degree)
                visualize_curve(control_points, curve_points, "B-Spline Curve")
                
        elif choice == '3':
            break
            
        else:
            print("Invalid choice!")

def transformations_3d_menu():
    # Default cube vertices
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    edges = [(0,1), (1,2), (2,3), (3,0),
             (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    
    while True:
        print("\n3D TRANSFORMATIONS")
        print("1. Translation")
        print("2. Scaling")
        print("3. Rotation (X-axis)")
        print("4. Rotation (Y-axis)")
        print("5. Rotation (Z-axis)")
        print("6. Perspective Projection")
        print("7. Back to Main Menu")
        
        choice = input("Enter choice (1-7): ")
        
        if choice == '1':
            tx = float(input("Enter tx: "))
            ty = float(input("Enter ty: "))
            tz = float(input("Enter tz: "))
            transformed = translate_3d(vertices, tx, ty, tz)
            visualize_3d(transformed, edges, "3D Translation")
            
        elif choice == '2':
            sx = float(input("Enter sx: "))
            sy = float(input("Enter sy: "))
            sz = float(input("Enter sz: "))
            px = float(input("Pivot x (default 0.5): ") or 0.5)
            py = float(input("Pivot y (default 0.5): ") or 0.5)
            pz = float(input("Pivot z (default 0.5): ") or 0.5)
            transformed = scale_3d(vertices, sx, sy, sz, (px, py, pz))
            visualize_3d(transformed, edges, "3D Scaling")
            
        elif choice in ['3', '4', '5']:
            angle = float(input("Enter angle (degrees): "))
            px = float(input("Pivot x (default 0.5): ") or 0.5)
            py = float(input("Pivot y (default 0.5): ") or 0.5)
            pz = float(input("Pivot z (default 0.5): ") or 0.5)
            
            if choice == '3':
                transformed = rotate_x_3d(vertices, angle, (px, py, pz))
                title = "Rotation Around X-axis"
            elif choice == '4':
                transformed = rotate_y_3d(vertices, angle, (px, py, pz))
                title = "Rotation Around Y-axis"
            else:
                transformed = rotate_z_3d(vertices, angle, (px, py, pz))
                title = "Rotation Around Z-axis"
                
            visualize_3d(transformed, edges, title)
            
        elif choice == '6':
            # Move cube away from camera
            transformed = translate_3d(vertices, 0, 0, 5)
            projected = perspective_projection(transformed)
            
            plt.figure(figsize=(8, 8))
            plt.scatter(projected[:, 0], projected[:, 1], s=50)
            plt.title("Perspective Projection")
            plt.grid(True)
            plt.axis('equal')
            plt.show()
            
        elif choice == '7':
            break
            
        else:
            print("Invalid choice!")

# =================================================================
# PROGRAM ENTRY POINT
# =================================================================

if __name__ == "__main__":
    main()