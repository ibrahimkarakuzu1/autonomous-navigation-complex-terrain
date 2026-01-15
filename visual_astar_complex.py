import heapq 
import math
import numpy as np
import matplotlib.pyplot as plt

MASS = 900.0
GRAVITY = 3.721
FRICTION = 0.04

def calculate_energy_cost(p1, p2):
    dist_2d = math.sqrt((p2[0] -p1[0])**2 + (p2[1] - p1[1])**2)
    delta_h = p2[2] - p1[2]
    dist_3d = math.sqrt(dist_2d**2 + delta_h**2)

    if dist_3d == 0: return 0
    theta_rad = math.atan2(delta_h, dist_2d)

    gravity_force = MASS * GRAVITY * math.sin(theta_rad)
    friction_force = MASS * GRAVITY * math.cos(theta_rad) * FRICTION
    total_force = gravity_force + friction_force

    if total_force < 0: total_force = 0
    return total_force*dist_3d

#karmaşık arazi motoru 
# rastgele tepeler ve çukurlarla gerçeğe benzer bir mars yüzeyi oluşturacağız

def generate_complex_terrain(size = 50, num_hills=15): 
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    X, Y = np.meshgrid(x,y)
    Z = np.zeros((size, size))

    np.random.seed(42)#her seferinde aynı harita çıkması için
    print(f"{num_hills} adet tepe veya krater oluşturuluyyor")

    for _ in range(num_hills):
        #rastgele konum ve yükseklik seç
        x0 = np.random.uniform(0, size)
        y0 = np.random.uniform(0, size)
        height = np.random.uniform(20,60)#20-60 m tepeler
        spread = np.random.uniform(3, 10)#tepelerin genişlik

        #gaussian tepe ekleme
        Z += height * np.exp(-1 * (((X - x0)**2 + (Y - y0)**2) / (2 * spread**2)))
        
    return Z

# A* algorithm
class Node: 
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.g = float('inf')
        self.f = float('inf')
        self.parent = None
    def __lt__(self, other): return self.f < other.f

def heuristic(node, goal):
    return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2 + (node.z - goal.z)**2 ) 

def reconstruct_path(current_node):
    path = []
    energy = current_node.g
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent
    return path[::-1], energy

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    start_node = Node(start[0], start[1], grid[start])
    goal_node = Node(goal[0], goal[1], grid[goal])
    start_node.g = 0
    start_node.f = heuristic(start_node, goal_node)

    open_list = []
    heapq.heappush(open_list, start_node)

    visited = set()
    nodes = {start: start_node}

    while open_list:
        current = heapq.heappop(open_list)
        if (current.x, current.y) == goal: return reconstruct_path(current)
        visited.add((current.x, current.y))

        for dx, dy in [(0,1), (0,-1),(1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) in visited: continue
                nz = grid[nx, ny]

                move_cost = calculate_energy_cost ((current.x, current.y, current.z), (nx, ny, nz))

                new_g = current.g + move_cost

                if (nx, ny) not in nodes or new_g < nodes[(nx, ny)].g:
                    neighbor = Node(nx, ny, nz)
                    neighbor.g = new_g
                    neighbor.f = new_g + heuristic(neighbor, goal_node)
                    neighbor.parent = current 
                    nodes[(nx, ny)] = neighbor
                    heapq.heappush(open_list, neighbor)
    return None, 0
if __name__ == "__main__":
    #karmasşık arazi
    GRID_SIZE = 60
    terrain = generate_complex_terrain(GRID_SIZE, num_hills = 140)

    start_pos = (0, 0)
    goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)

    path, total_energy = a_star_search(terrain,start_pos, goal_pos)

    if path:
        print(f" Rota Bulundu! Toplam Enerji: {total_energy:.2f} J")
        
        plt.figure(figsize=(10, 8))
        # 'terrain' yerine 'gist_earth' renk haritası kullanalım (Daha gerçekçi)
        plt.imshow(terrain, cmap='gist_earth', origin='lower')
        plt.colorbar(label='Yükseklik (m)')
        
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        
        # Rotayı daha belirgin çizelim (Sarı renk)
        plt.plot(path_x, path_y, color='yellow', linewidth=2, label='Optimum Rota')
        plt.scatter(start_pos[1], start_pos[0], c='lime', s=100, edgecolors='k', label='Start')
        plt.scatter(goal_pos[1], goal_pos[0], c='red', s=100, marker='*', edgecolors='k', label='Goal')
        
        plt.title(f"Kaotik Arazide Navigasyon\nEnerji Tüketimi: {total_energy:.0f} Joule")
        plt.legend()
        plt.show()
    else:
        print(" Yol bulunamadı!")
