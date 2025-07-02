import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# ---------------------
# Cohen-Sutherland Line Clipping
# ---------------------
# Region codes
INSIDE = 0  # 0000
LEFT   = 1  # 0001
RIGHT  = 2  # 0010
BOTTOM = 4  # 0100
TOP    = 8  # 1000

def compute_outcode(x, y, xmin, ymin, xmax, ymax):
    code = INSIDE
    if x < xmin:   code |= LEFT
    elif x > xmax: code |= RIGHT
    if y < ymin:   code |= BOTTOM
    elif y > ymax: code |= TOP
    return code

def cohen_sutherland_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    outcode0 = compute_outcode(x0, y0, xmin, ymin, xmax, ymax)
    outcode1 = compute_outcode(x1, y1, xmin, ymin, xmax, ymax)
    accept = False

    while True:
        # Trivial accept
        if outcode0 == 0 and outcode1 == 0:
            accept = True
            break
        # Trivial reject
        elif (outcode0 & outcode1) != 0:
            break
        else:
            # Choose one endpoint outside
            outcode_out = outcode0 if outcode0 != 0 else outcode1
            # Find intersection
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
            # Replace point outside
            if outcode_out == outcode0:
                x0, y0 = x, y
                outcode0 = compute_outcode(x0, y0, xmin, ymin, xmax, ymax)
            else:
                x1, y1 = x, y
                outcode1 = compute_outcode(x1, y1, xmin, ymin, xmax, ymax)

    if accept:
        return (x0, y0), (x1, y1)
    else:
        return None

# ---------------------
# Clipping Menu
# ---------------------
def clipping_menu():
    print("\n--- Cohen-Sutherland Line Clipping ---")
    xmin, ymin = map(float, input("Enter clip window min corner (xmin ymin): ").split())
    xmax, ymax = map(float, input("Enter clip window max corner (xmax ymax): ").split())
    x0, y0 = map(float, input("Enter line start point (x0 y0): ").split())
    x1, y1 = map(float, input("Enter line end point (x1 y1): ").split())

    result = cohen_sutherland_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax)
    if result:
        (cx0, cy0), (cx1, cy1) = result
        print(f"Clipped line from ({cx0:.2f}, {cy0:.2f}) to ({cx1:.2f}, {cy1:.2f})")
        visualize_clipped_line((x0, y0), (x1, y1), (cx0, cy0), (cx1, cy1), (xmin, ymin, xmax, ymax))
        print("\nVisualization legend:")
        print(f"  Green rectangle: Clip window ({xmin},{ymin}) to ({xmax},{ymax})")
        print(f"  Blue dashed line: Original line from ({x0},{y0}) to ({x1},{y1})")
        print("  Red solid line: Clipped portion inside the window")
        print("  Red circles: Clipped endpoints")
        print("\nAlgorithm overview:")
        print("  • Assign region codes to endpoints")
        print("  • Quick accept/reject trivial cases")
        print("  • Calculate intersections for crossing lines")
        print("  • Iterate until accept or reject")
    else:
        print("Line rejected (outside clipping window)")
    input("Press Enter to continue...")

# ---------------------
# Visualization for Clipping
# ---------------------
def visualize_clipped_line(p0, p1, c0, c1, window):
    xmin, ymin, xmax, ymax = window
    fig, ax = plt.subplots()
    # Original line (dashed blue)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linestyle='--', color='blue', label='Original Line')
    # Clipped line (solid red)
    ax.plot([c0[0], c1[0]], [c0[1], c1[1]], linestyle='-', linewidth=2, color='red', label='Clipped Line')
    # Clipped endpoints
    ax.scatter([c0[0], c1[0]], [c0[1], c1[1]], edgecolors='red', facecolors='none', s=50, linewidths=2)
    # Clipping window (green)
    rect_x = [xmin, xmax, xmax, xmin, xmin]
    rect_y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(rect_x, rect_y, color='green', label='Clip Window')

    ax.set_title('Cohen-Sutherland Clipping')
    ax.set_aspect('equal'); ax.legend(); ax.grid(True)
    plt.show()

# ---------------------
# Main Program
# ---------------------
def main():
    while True:
        print("\nComputer Graphics Algorithms")
        print("1. 2D Transformations")
        print("2. Line Drawing Algorithms")
        print("3. Circle Drawing Algorithm")
        print("4. Line Clipping (Cohen-Sutherland)")
        print("5. Polygon Clipping (Sutherland-Hodgman)")
        print("6. Area Filling (Scanline Algorithm)")
        print("7. Spline Curves (Bezier)")
        print("8. Exit")

        choice = input("Enter choice (1-8): ")

        if choice == '1':
            transformations_menu()
        elif choice == '2':
            line_drawing_menu()
        elif choice == '3':
            circle_drawing_menu()
        elif choice == '4':
            clipping_menu()
        elif choice == '5':
            polygon_clipping_menu()
        elif choice == '6':
            filling_menu()
        elif choice == '7':
            spline_menu()
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

# ---------------------
# Menus
# ---------------------
def transformations_menu():
    points = np.array([[0, 0], [50, 100], [100, 0]])  # Triangle
    while True:
        print("\n2D Transformations")
        print("1. Translate")
        print("2. Scale")
        print("3. Rotate")
        print("4. Combined Transformations")
        print("5. Back")

        choice = input("Enter choice (1-5): ")
        if choice == '1':
            tx = float(input("Enter tx: "))
            ty = float(input("Enter ty: "))
            transformed = translate(points, tx, ty)
            visualize_points(points, transformed, "Translation")
        elif choice == '2':
            sx = float(input("Enter sx: "))
            sy = float(input("Enter sy: "))
            transformed = scale(points, sx, sy)
            visualize_points(points, transformed, "Scaling")
        elif choice == '3':
            angle = float(input("Enter angle (degrees): "))
            px = float(input("Pivot x (default 0): ") or 0)
            py = float(input("Pivot y (default 0): ") or 0)
            transformed = rotate(points, angle, (px, py))
            visualize_points(points, transformed, "Rotation")
        elif choice == '4':
            tx = float(input("Enter tx for translation: "))
            ty = float(input("Enter ty for translation: "))
            sx = float(input("Enter sx for scaling: "))
            sy = float(input("Enter sy for scaling: "))
            angle = float(input("Enter angle for rotation (degrees): "))
            px = float(input("Rotation pivot x (default 0): ") or 0)
            py = float(input("Rotation pivot y (default 0): ") or 0)
            p1 = translate(points, tx, ty)
            p2 = rotate(p1, angle, (px, py))
            transformed = scale(p2, sx, sy)
            visualize_points(points, transformed, "Combined Transformations")
        elif choice == '5':
            break
        else:
            print("Invalid choice!")

def line_drawing_menu():
    while True:
        print("\nLine Drawing Algorithms")
        print("1. Bresenham's Algorithm")
        print("2. DDA Algorithm")
        print("3. Back")

        choice = input("Enter choice (1-3): ")
        if choice == '1':
            x0, y0 = map(int, input("Enter start point (x0 y0): ").split())
            x1, y1 = map(int, input("Enter end point (x1 y1): ").split())
            points, table = bresenham_line_with_table(x0, y0, x1, y1)
            print(table)
            visualize_line(points, "Bresenham's Line")
        elif choice == '2':
            x0, y0 = map(int, input("Enter start point (x0 y0): ").split())
            x1, y1 = map(int, input("Enter end point (x1 y1): ").split())
            points, table = dda_line_with_table(x0, y0, x1, y1)
            print(table)
            visualize_line(points, "DDA Line")
        elif choice == '3':
            break
        else:
            print("Invalid choice!")

def circle_drawing_menu():
    while True:
        print("\nCircle Drawing")
        print("1. Midpoint Algorithm")
        print("2. Back")

        choice = input("Enter choice (1-2): ")
        if choice == '1':
            r = int(input("Enter radius: "))
            cx, cy = map(int, input("Enter center (cx cy): ").split())
            points, table = midpoint_circle_with_table(r, (cx, cy))
            print(table)
            visualize_circle(points, "Midpoint Circle")
        elif choice == '2':
            break
        else:
            print("Invalid choice!")

def polygon_clipping_menu():
    print("\n--- Sutherland–Hodgman Polygon Clipping ---")
    xmin, ymin = map(float, input("Enter clip window min corner (xmin ymin): ").split())
    xmax, ymax = map(float, input("Enter clip window max corner (xmax ymax): ").split())

    n = int(input("How many vertices in the subject polygon? "))
    subject = [tuple(map(float, input(f" Vertex {i+1} (x y): ").split())) for i in range(n)]

    def clip_edge(poly, inside_fn, intersect_fn):
        output = []
        if not poly:
            return output
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

    def inside_left(p):   return p[0] >= xmin
    def intersect_left(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        t = (xmin - x1) / (x2 - x1)
        return (xmin, y1 + t*(y2 - y1))

    def inside_right(p):  return p[0] <= xmax
    def intersect_right(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        t = (xmax - x1) / (x2 - x1)
        return (xmax, y1 + t*(y2 - y1))

    def inside_bottom(p): return p[1] >= ymin
    def intersect_bottom(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        t = (ymin - y1) / (y2 - y1)
        return (x1 + t*(x2 - x1), ymin)

    def inside_top(p):    return p[1] <= ymax
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

    if clipped:
        print("\nClipped polygon vertices:")
        for x, y in clipped:
            print(f" ({x:.2f}, {y:.2f})")
        visualize_polygon(subject, clipped, (xmin, ymin, xmax, ymax))
    else:
        print("\nPolygon completely outside the clipping window.")
    input("Press Enter to continue...")

def filling_menu():
    print("\n--- Scanline Polygon Filling ---")
    n = int(input("How many vertices in the polygon? "))
    polygon = [tuple(map(float, input(f" Vertex {i+1} (x y): ").split())) for i in range(n)]
    
    filled_points = scanline_fill(polygon)
    
    visualize_fill(polygon, filled_points)
    input("Press Enter to continue...")

def spline_menu():
    print("\n--- Bezier Curve ---")
    print("Enter 4 control points")
    control_points = []
    for i in range(4):
        x, y = map(float, input(f"Control point {i+1} (x y): ").split())
        control_points.append((x, y))
    
    curve_points = bezier_curve(control_points)
    
    visualize_bezier(control_points, curve_points)
    input("Press Enter to continue...")

# ---------------------
# Visualization Functions
# ---------------------
def visualize_points(original, transformed, title):
    fig, ax = plt.subplots()
    x_orig, y_orig = zip(*original)
    ax.plot(x_orig, y_orig, 'bo-', label='Original')
    x_trans, y_trans = zip(*transformed)
    ax.plot(x_trans, y_trans, 'ro-', label='Transformed')
    if len(original) >= 3:
        ax.plot([x_orig[0], x_orig[-1]], [y_orig[0], y_orig[-1]], 'b--')
        ax.plot([x_trans[0], x_trans[-1]], [y_trans[0], y_trans[-1]], 'r--')
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

def visualize_polygon(subject, clipped, window):
    xmin, ymin, xmax, ymax = window
    fig, ax = plt.subplots()
    xs, ys = zip(*(subject + [subject[0]]))
    ax.plot(xs, ys, 'b--', label='Subject')
    xs, ys = zip(*(clipped + [clipped[0]]))
    ax.plot(xs, ys, 'r-', linewidth=2, label='Clipped')
    rect_x = [xmin, xmax, xmax, xmin, xmin]
    rect_y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(rect_x, rect_y, 'g-', label='Clip window')
    ax.set_aspect('equal'); ax.legend(); ax.grid(True)
    plt.show()

def visualize_fill(polygon, filled_points):
    fig, ax = plt.subplots()
    xs, ys = zip(*(polygon + [polygon[0]]))
    ax.plot(xs, ys, 'b-', label='Polygon')
    if filled_points:
        x_vals, y_vals = zip(*filled_points)
        ax.scatter(x_vals, y_vals, color='red', s=5, label='Filled Pixels')
    ax.set_title("Scanline Fill Algorithm")
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    plt.show()

def visualize_bezier(control_points, curve_points):
    fig, ax = plt.subplots()
    cx, cy = zip(*control_points)
    ax.plot(cx, cy, 'bo-', label='Control Polygon')
    if curve_points:
        bx, by = zip(*curve_points)
        ax.plot(bx, by, 'r-', linewidth=2, label='Bezier Curve')
    ax.set_title("Bezier Curve")
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    plt.show()

# ---------------------
# Line Drawing Algorithms
# ---------------------
def bresenham_line_with_table(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    decixion_table = "Step  x  y  Decision\n"
    
    step = 0
    while True:
        points.append((x0, y0))
        decixion_table += f"{step:4}  {x0:2}  {y0:2}  {err:4}\n"
        
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
        step += 1
        
    return points, decixion_table

def dda_line_with_table(x0, y0, x1, y1):
    points = []
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(x0, y0)], "Single Point"
    
    x_inc = dx / steps
    y_inc = dy / steps
    x = x0
    y = y0
    
    table = "Step  x       y       Rounded\n"
    for i in range(steps + 1):
        points.append((round(x), round(y)))
        table += f"{i:4}  {x:.2f}    {y:.2f}    ({round(x)}, {round(y)})\n"
        x += x_inc
        y += y_inc
        
    return points, table

def visualize_line(points, title):
    fig, ax = plt.subplots()
    x_vals, y_vals = zip(*points)
    ax.plot(x_vals, y_vals, 'bo-')
    for i, (x, y) in enumerate(points):
        ax.text(x, y, f'({x},{y})', fontsize=8)
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()

# ---------------------
# Circle Drawing Algorithm
# ---------------------
def midpoint_circle_with_table(radius, center=(0, 0)):
    cx, cy = center
    x = radius
    y = 0
    decision = 1 - radius
    points = []
    table = "  x   y  Decision\n"
    
    def add_points(x, y, cx, cy):
        return [
            (cx + x, cy + y),
            (cx - x, cy + y),
            (cx + x, cy - y),
            (cx - x, cy - y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx + y, cy - x),
            (cx - y, cy - x)
        ]
    
    step = 0
    while x >= y:
        new_points = add_points(x, y, cx, cy)
        points.extend(new_points)
        table += f"{x:3} {y:3} {decision:5}\n"
        
        y += 1
        if decision <= 0:
            decision = decision + 2 * y + 1
        else:
            x -= 1
            decision = decision + 2 * (y - x) + 1
        step += 1
        
    return points, table

def visualize_circle(points, title):
    fig, ax = plt.subplots()
    x_vals, y_vals = zip(*points)
    ax.scatter(x_vals, y_vals, s=5)
    if points:
        cx = sum(x for x, y in points) / len(points)
        cy = sum(y for x, y in points) / len(points)
        ax.plot(cx, cy, 'ro', markersize=3)
        ax.text(cx, cy, 'Center', fontsize=8)
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()

# ---------------------
# Transformations
# ---------------------
def translate(points, tx, ty):
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    homog = np.hstack([points, np.ones((points.shape[0],1))])
    print("\nTranslation Matrix:\n", T)
    return (homog @ T.T)[:,:2]

def scale(points, sx, sy, fixed=(0, 0)):
    T1 = np.array([[1,0,-fixed[0]],[0,1,-fixed[1]],[0,0,1]])
    S  = np.array([[sx,0,0],[0,sy,0],[0,0,1]])
    T2 = np.array([[1,0,fixed[0]],[0,1,fixed[1]],[0,0,1]])
    print("\nScaling Matrices:\nT1:\n",T1,"\nS:\n",S,"\nT2:\n",T2)
    homog = np.hstack([points, np.ones((points.shape[0],1))])
    step1 = (homog @ T1.T)[:,:2]
    homog2 = np.hstack([step1, np.ones((step1.shape[0],1))])
    step2 = (homog2 @ S.T)[:,:2]
    homog3 = np.hstack([step2, np.ones((step2.shape[0],1))])
    final = (homog3 @ T2.T)[:,:2]
    return final

def rotate(points, angle, pivot=(0, 0)):
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    T1 = np.array([[1,0,-pivot[0]],[0,1,-pivot[1]],[0,0,1]])
    R  = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    T2 = np.array([[1,0,pivot[0]],[0,1,pivot[1]],[0,0,1]])
    print("\nRotation Matrices:\nT1:\n",T1,"\nR:\n",R,"\nT2:\n",T2)
    homog = np.hstack([points, np.ones((points.shape[0],1))])
    step1 = (homog @ T1.T)[:,:2]
    homog2 = np.hstack([step1, np.ones((step1.shape[0],1))])
    step2 = (homog2 @ R.T)[:,:2]
    homog3 = np.hstack([step2, np.ones((step2.shape[0],1))])
    final = (homog3 @ T2.T)[:,:2]
    return final

# ---------------------
# Area Filling (Scanline)
# ---------------------
def scanline_fill(polygon):
    if not polygon:
        return []
    y_min = int(min(y for x, y in polygon))
    y_max = int(max(y for x, y in polygon))
    filled_points = []
    edges = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
        if y1 == y2:
            continue
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        inv_slope = (x2 - x1) / (y2 - y1)
        edges.append((y1, y2, x1, inv_slope))
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
                x_end = int(np.floor(intersections[i+1]))
                for x in range(x_start, x_end + 1):
                    filled_points.append((x, y))
    return filled_points

# ---------------------
# Spline Curves (Bezier)
# ---------------------
def bezier_curve(control_points, num_points=100):
    if len(control_points) != 4:
        raise ValueError("Bezier curve requires exactly 4 control points")
    curve = []
    for i in range(num_points + 1):
        t = i / num_points
        x = (1-t)**3 * control_points[0][0] + 3*(1-t)**2*t*control_points[1][0] + 3*(1-t)*t**2*control_points[2][0] + t**3*control_points[3][0]
        y = (1-t)**3 * control_points[0][1] + 3*(1-t)**2*t*control_points[1][1] + 3*(1-t)*t**2*control_points[2][1] + t**3*control_points[3][1]
        curve.append((x, y))
    return curve

if __name__ == '__main__':
    main()