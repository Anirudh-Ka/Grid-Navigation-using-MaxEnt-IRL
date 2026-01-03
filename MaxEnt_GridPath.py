import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import random
from collections import defaultdict

class NavigationWorld:
    def __init__(self, world_size=10, num_obstacles=5):
        self.size = world_size
        self.start_position = np.array([1.0, 1.0])
        self.goal_position = np.array([8.5, 8.5])
        self.obstacles = self._place_random_obstacles(num_obstacles)
        self.grid_resolution = 0.5
        
    def _place_random_obstacles(self, count):
        obstacles = []
        
        for _ in range(count // 2):
            while True:
                center = np.random.uniform(2, self.size - 2, 2)
                radius = np.random.uniform(0.3, 0.8)
                
                if (np.linalg.norm(center - self.start_position) > radius + 1.0 and 
                    np.linalg.norm(center - self.goal_position) > radius + 1.0):
                    obstacles.append({'shape': 'circle', 'center': center, 'radius': radius})
                    break
        
        for _ in range(count - count // 2):
            while True:
                position = np.random.uniform(2, self.size - 3, 2)
                size = np.random.uniform(0.5, 1.5, 2)
                rect_center = position + size / 2
                
                if (np.linalg.norm(rect_center - self.start_position) > 1.5 and 
                    np.linalg.norm(rect_center - self.goal_position) > 1.5):
                    obstacles.append({'shape': 'rectangle', 'position': position, 'size': size})
                    break
        
        return obstacles
    
    def would_collide(self, point):
        for obstacle in self.obstacles:
            if obstacle['shape'] == 'circle':
                distance = np.linalg.norm(point - obstacle['center'])
                if distance < obstacle['radius']:
                    return True
            elif obstacle['shape'] == 'rectangle':
                pos, size = obstacle['position'], obstacle['size']
                if (pos[0] <= point[0] <= pos[0] + size[0] and 
                    pos[1] <= point[1] <= pos[1] + size[1]):
                    return True
        return False
    
    def analyze_path(self, path):
        if len(path) < 2:
            return np.zeros(5)
        
        path = np.array(path)
        
        total_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path) - 1))
        
        total_curvature = 0
        if len(path) >= 3:
            for i in range(1, len(path) - 1):
                vector_before = path[i] - path[i-1]
                vector_after = path[i+1] - path[i]
                
                norm_before = np.linalg.norm(vector_before)
                norm_after = np.linalg.norm(vector_after)
                
                if norm_before > 0 and norm_after > 0:
                    cos_angle = np.dot(vector_before, vector_after) / (norm_before * norm_after)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    total_curvature += angle
        
        collision_count = sum(1 for point in path if self.would_collide(point))
        
        closest_approach = float('inf')
        for point in path:
            for obstacle in self.obstacles:
                if obstacle['shape'] == 'circle':
                    distance = np.linalg.norm(point - obstacle['center']) - obstacle['radius']
                    closest_approach = min(closest_approach, max(0, distance))
                elif obstacle['shape'] == 'rectangle':
                    pos, size = obstacle['position'], obstacle['size']
                    rect_center = pos + size / 2
                    distance = np.linalg.norm(point - rect_center) - np.linalg.norm(size) / 2
                    closest_approach = min(closest_approach, max(0, distance))
        
        straight_line_distance = np.linalg.norm(self.goal_position - self.start_position)
        final_distance = np.linalg.norm(path[-1] - self.goal_position)
        progress = straight_line_distance - final_distance
        
        features = np.array([
            total_length / self.size,
            total_curvature / (len(path) - 1) if len(path) > 1 else 0,
            collision_count / len(path),
            1.0 / (closest_approach + 0.1),
            progress / straight_line_distance
        ])
        
        return features
    
    def draw(self, ax=None, paths=None, show_grid=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        
        if show_grid:
            for i in np.arange(0, self.size, self.grid_resolution):
                ax.axhline(i, color='gray', linewidth=0.2, alpha=0.3)
                ax.axvline(i, color='gray', linewidth=0.2, alpha=0.3)
        
        for obstacle in self.obstacles:
            if obstacle['shape'] == 'circle':
                circle = Circle(obstacle['center'], obstacle['radius'], 
                              color='red', alpha=0.3, edgecolor='darkred', linewidth=2)
                ax.add_patch(circle)
            elif obstacle['shape'] == 'rectangle':
                rect = Rectangle(obstacle['position'], obstacle['size'][0], obstacle['size'][1],
                               color='red', alpha=0.3, edgecolor='darkred', linewidth=2)
                ax.add_patch(rect)
        
        ax.plot(self.start_position[0], self.start_position[1], 'go', markersize=15, 
                label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(self.goal_position[0], self.goal_position[1], 'b*', markersize=20, 
                label='Goal', markeredgecolor='darkblue', markeredgewidth=2)
        
        if paths is not None:
            for i, path in enumerate(paths):
                path_array = np.array(path)
                opacity = 0.3 + 0.5 * (i / len(paths))
                ax.plot(path_array[:, 0], path_array[:, 1], 'b-', alpha=opacity, linewidth=2)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.2)
        return ax


class ExpertNavigator:
    def __init__(self, world):
        self.world = world
    
    def _estimate_distance(self, position, goal):
        return np.linalg.norm(position - goal)
    
    def _find_neighbors(self, position):
        step = self.world.grid_resolution
        directions = [
            (step, 0), (-step, 0), (0, step), (0, -step),
            (step, step), (-step, -step), (step, -step), (-step, step)
        ]
        
        neighbors = []
        for dx, dy in directions:
            new_position = position + np.array([dx, dy])
            
            if (0 <= new_position[0] <= self.world.size and 
                0 <= new_position[1] <= self.world.size and
                not self.world.would_collide(new_position)):
                neighbors.append(new_position)
        
        return neighbors
    
    def _smooth_path(self, path, iterations=5):
        if len(path) < 3:
            return path
        
        smoothed = np.array(path).copy()
        
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                smoothed[i] = 0.5 * smoothed[i] + 0.25 * (smoothed[i-1] + smoothed[i+1])
                
                if self.world.would_collide(smoothed[i]):
                    smoothed[i] = path[i]
        
        return smoothed.tolist()
    
    def find_path(self):
        start = tuple(self.world.start_position)
        goal = tuple(self.world.goal_position)
        
        open_set = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while open_set:
            _, current = min(open_set, key=lambda x: x[0])
            open_set.remove((_, current))
            
            if np.linalg.norm(np.array(current) - self.world.goal_position) < self.world.grid_resolution:
                break
            
            for next_position in self._find_neighbors(np.array(current)):
                next_tuple = tuple(next_position)
                step_cost = np.linalg.norm(next_position - np.array(current))
                new_cost = cost_so_far[current] + step_cost
                
                if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                    cost_so_far[next_tuple] = new_cost
                    priority = new_cost + self._estimate_distance(next_position, self.world.goal_position)
                    open_set.append((priority, next_tuple))
                    came_from[next_tuple] = current
        
        path = []
        current = tuple(self.world.goal_position)
        
        if current not in came_from:
            current = min(came_from.keys(), 
                         key=lambda p: np.linalg.norm(np.array(p) - self.world.goal_position))
        
        while current is not None:
            path.append(np.array(current))
            current = came_from.get(current)
        
        path.reverse()
        return self._smooth_path(path, iterations=3)


class RewardLearner:
    def __init__(self, world, num_features=5, learning_rate=0.01, discount=0.95):
        self.world = world
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.discount = discount
        self.feature_weights = np.random.randn(num_features) * 0.1
        self.valid_positions = self._build_position_grid()
        
        print("Initialized reward learner with these features:")
        print("  • Path length")
        print("  • Curvature")
        print("  • Collisions")
        print("  • Distance to obstacles")
        print("  • Progress toward goal")
    
    def _build_position_grid(self):
        positions = []
        step = self.world.grid_resolution
        
        for x in np.arange(0, self.world.size + step, step):
            for y in np.arange(0, self.world.size + step, step):
                position = np.array([x, y])
                if not self.world.would_collide(position):
                    positions.append(position)
        
        return positions
    
    def _get_position_features(self, position):
        goal_distance = np.linalg.norm(position - self.world.goal_position) / self.world.size
        
        nearest_obstacle = float('inf')
        for obstacle in self.world.obstacles:
            if obstacle['shape'] == 'circle':
                distance = np.linalg.norm(position - obstacle['center']) - obstacle['radius']
            else:
                pos, size = obstacle['position'], obstacle['size']
                center = pos + size / 2
                distance = np.linalg.norm(position - center) - np.linalg.norm(size) / 2
            nearest_obstacle = min(nearest_obstacle, max(0, distance))
        
        in_collision = 1.0 if self.world.would_collide(position) else 0.0
        
        total_distance = np.linalg.norm(self.world.goal_position - self.world.start_position)
        remaining = np.linalg.norm(position - self.world.goal_position)
        progress = 1.0 - (remaining / total_distance)
        
        features = np.array([
            0.1,
            0.0,
            in_collision * 10.0,
            1.0 / (nearest_obstacle + 0.1),
            progress
        ])
        
        return features
    
    def _calculate_reward(self, position):
        features = self._get_position_features(position)
        return np.dot(self.feature_weights, features)
    
    def _get_neighbors(self, position):
        step = self.world.grid_resolution
        directions = [
            (step, 0), (-step, 0), (0, step), (0, -step),
            (step, step), (-step, -step), (step, -step), (-step, step)
        ]
        
        neighbors = []
        for dx, dy in directions:
            next_pos = position + np.array([dx, dy])
            if (0 <= next_pos[0] <= self.world.size and 
                0 <= next_pos[1] <= self.world.size):
                neighbors.append(next_pos)
        
        return neighbors
    
    def _compute_soft_values(self, max_iterations=50):
        values = defaultdict(float)
        
        for iteration in range(max_iterations):
            new_values = defaultdict(float)
            
            for position in self.valid_positions:
                pos_tuple = tuple(position)
                
                if np.linalg.norm(position - self.world.goal_position) < self.world.grid_resolution:
                    new_values[pos_tuple] = 0.0
                    continue
                
                q_values = []
                for next_position in self._get_neighbors(position):
                    next_tuple = tuple(next_position)
                    reward = self._calculate_reward(position)
                    q_value = reward + self.discount * values[next_tuple]
                    q_values.append(q_value)
                
                if q_values:
                    max_q = max(q_values)
                    new_values[pos_tuple] = max_q + np.log(sum(np.exp(q - max_q) for q in q_values))
                else:
                    new_values[pos_tuple] = 0.0
            
            values = new_values
        
        return values
    
    def _compute_state_frequencies(self, values, expert_paths):
        frequencies = defaultdict(float)
        frequencies[tuple(self.world.start_position)] = 1.0
        total_frequencies = defaultdict(float)
        
        for time_step in range(50):
            new_frequencies = defaultdict(float)
            
            for pos_tuple, frequency in frequencies.items():
                position = np.array(pos_tuple)
                
                if np.linalg.norm(position - self.world.goal_position) < self.world.grid_resolution:
                    continue
                
                neighbors = self._get_neighbors(position)
                if not neighbors:
                    continue
                
                q_values = []
                for next_position in neighbors:
                    next_tuple = tuple(next_position)
                    reward = self._calculate_reward(position)
                    q_value = reward + self.discount * values[next_tuple]
                    q_values.append((next_position, q_value))
                
                max_q = max(q for _, q in q_values)
                exp_values = [np.exp(q - max_q) for _, q in q_values]
                total_exp = sum(exp_values)
                probabilities = [e / total_exp for e in exp_values]
                
                for (next_position, _), probability in zip(q_values, probabilities):
                    next_tuple = tuple(next_position)
                    new_frequencies[next_tuple] += frequency * probability * (self.discount ** time_step)
                
                total_frequencies[pos_tuple] += frequency * (self.discount ** time_step)
            
            frequencies = new_frequencies
        
        return total_frequencies
    
    def _compute_expected_features(self, state_frequencies):
        expected = np.zeros(self.num_features)
        
        for pos_tuple, frequency in state_frequencies.items():
            position = np.array(pos_tuple)
            features = self._get_position_features(position)
            expected += frequency * features
        
        return expected
    
    def evaluate_path(self, path):
        total_reward = 0
        for position in path:
            position = np.array(position) if not isinstance(position, np.ndarray) else position
            total_reward += self._calculate_reward(position)
        return total_reward
    
    def learn_from_experts(self, expert_paths, iterations=100):
        print(f"\nLearning from {len(expert_paths)} expert demonstrations...")
        
        expert_frequencies = defaultdict(float)
        total_positions = 0
        
        for path in expert_paths:
            for position in path:
                position = np.array(position) if not isinstance(position, np.ndarray) else position
                rounded_pos = np.round(position / self.world.grid_resolution) * self.world.grid_resolution
                pos_tuple = tuple(rounded_pos)
                expert_frequencies[pos_tuple] += 1.0
                total_positions += 1
        
        for pos_tuple in expert_frequencies:
            expert_frequencies[pos_tuple] /= total_positions
        
        expert_features = self._compute_expected_features(expert_frequencies)
        print(f"\nExpert feature profile: {expert_features}")
        
        loss_history = []
        weight_history = [self.feature_weights.copy()]
        
        for iteration in range(iterations):
            values = self._compute_soft_values(max_iterations=30)
            learner_frequencies = self._compute_state_frequencies(values, expert_paths)
            learner_features = self._compute_expected_features(learner_frequencies)
            
            gradient = expert_features - learner_features
            self.feature_weights += self.learning_rate * gradient
            weight_history.append(self.feature_weights.copy())
            
            loss = np.linalg.norm(gradient)
            loss_history.append(loss)
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Matching error = {loss:.4f}")
                print(f"  Current weights: {self.feature_weights}")
        
        print("\n✓ Learning complete!")
        print(f"Final learned weights: {self.feature_weights}")
        
        return loss_history, weight_history
    
    def visualize_results(self, expert_paths, losses, weight_history):
        fig = plt.figure(figsize=(18, 10))
        
        ax1 = plt.subplot(2, 3, 1)
        self.world.draw(ax=ax1, paths=expert_paths[:5])
        ax1.set_title('Expert Demonstrations\n(First 5 paths)', fontsize=12, fontweight='bold')
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(losses, linewidth=2, color='darkblue')
        ax2.set_title('Learning Progress', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Feature Matching Error')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(2, 3, 3)
        weight_history = np.array(weight_history)
        feature_names = ['Length', 'Curvature', 'Collision', 'Obstacle Dist', 'Progress']
        colors = ['red', 'blue', 'orange', 'green', 'purple']
        
        for i in range(weight_history.shape[1]):
            ax3.plot(weight_history[:, i], label=feature_names[i], 
                    linewidth=2, color=colors[i])
        ax3.set_title('Weight Evolution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Weight Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(2, 3, 4)
        final_weights = weight_history[-1]
        bars = ax4.bar(feature_names, final_weights, color=colors, 
                      alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_title('Final Learned Weights', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Weight Value')
        ax4.axhline(0, color='black', linewidth=0.8)
        ax4.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        for bar, weight in zip(bars, final_weights):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top', fontweight='bold')
        
        ax5 = plt.subplot(2, 3, 5)
        expert_features = np.zeros(self.num_features)
        for path in expert_paths:
            expert_features += self.world.analyze_path(path)
        expert_features /= len(expert_paths)
        
        bars = ax5.bar(feature_names, expert_features, color=colors, 
                      alpha=0.7, edgecolor='black', linewidth=2)
        ax5.set_title('Expert Feature Profile', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Average Feature Value')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        ax6 = plt.subplot(2, 3, 6)
        self.world.draw(ax=ax6)
        
        if expert_paths:
            sample_path = expert_paths[0]
            path_array = np.array(sample_path)
            ax6.plot(path_array[:, 0], path_array[:, 1], 'b-', 
                    linewidth=3, label='Sample Expert Path', alpha=0.8)
            
            reward = self.evaluate_path(sample_path)
            ax6.set_title(f'Sample Path (Reward: {reward:.2f})', 
                         fontsize=12, fontweight='bold')
            ax6.legend()
        
        plt.tight_layout()
        plt.savefig('learned_navigation.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved as 'learned_navigation.png'")
        plt.show()

def run_demonstration():
    print("=" * 70)
    print("  LEARNING TO NAVIGATE BY WATCHING EXPERTS")
    print("  (Inverse Reinforcement Learning Demo)")
    print("=" * 70)
    
    print("\nCreating world...")
    world = NavigationWorld(world_size=10, num_obstacles=6)
    print(f"  ✓ Created {world.size}x{world.size} world with {len(world.obstacles)} obstacles")
    
    print("\nGenerating expert demonstrations...")
    expert = ExpertNavigator(world)
    expert_paths = []
    
    for i in range(10):
        path = expert.find_path()
        if len(path) > 1:
            expert_paths.append(path)
            print(f"  ✓ Path {i+1}: {len(path)} waypoints")
    
    print(f"\n  Generated {len(expert_paths)} successful paths")
    
    print("\n Reward function from demonstrations...")
    learner = RewardLearner(world, num_features=5, learning_rate=0.1, discount=0.95)
    losses, weights = learner.learn_from_experts(expert_paths, iterations=100)
    
    print("\nVisualizations...")
    learner.visualize_results(expert_paths, losses, weights)
    
    print("\n" + "=" * 70)
    print("  LEARNED REWARD FUNCTION")
    print("=" * 70)
    
    feature_names = ['Path Length', 'Curvature', 'Collisions', 'Obstacle Distance', 'Goal Progress']
    
    for name, weight in zip(feature_names, learner.feature_weights):
        if weight > 0:
            print(f"  ↑ {name:20s}: {weight:+.3f}  (encouraged)")
        else:
            print(f"  ↓ {name:20s}: {weight:+.3f}  (avoided)")
    
    print("\n" + "=" * 70)
    print("  Done! Check 'learned_navigation.png' for detailed results.")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()