"""
Disaster Evacuation Planning using Informed Search - COMPLETE PROGRAM
Everything in one file:
  1. You type the number of people, exits, groups, exit capacity, disaster and strategy
  2. The program builds a building and plans the evacuation with A* (with re-planning)
  3. It prints the evacuation log and the results
  4. It compares two exit strategies (nearest exit vs balanced exits)
  5. It compares four search algorithms (A*, Dijkstra, BFS, Greedy)
  6. It makes an animation (GIF, and an inline player in Google Colab)

Install:  pip install networkx matplotlib pillow      (Google Colab already has them)
Run:      python evacuation_complete.py
Press Enter on any question to use the default value shown in [brackets].
"""
import heapq
import random
import time
from collections import deque

import networkx as nx
import matplotlib

try:                                      # inside Colab / Jupyter -> show animation inline
    get_ipython()                         # noqa: F821
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False
    if not matplotlib.get_backend().lower().startswith(("tk", "qt", "macosx")):
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors

ROWS, COLS = 12, 24
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
HAZARD_COLOR = {"Fire": "#e8590c", "Flood": "#1c7ed6", "Earthquake": "#8d6e63"}
GROUP_COLORS = ["#7048e8", "#f59f00", "#0ca678", "#d6336c", "#1098ad",
                "#e64980", "#5c940d", "#f76707", "#364fc7"]
ALGORITHMS = ["A*", "Dijkstra", "BFS", "Greedy"]


# =============================================================== 1. INPUT HELPERS
def ask_int(prompt, default, low, high):
    """Ask for a whole number between low and high (Enter = default)."""
    while True:
        text = input(f"{prompt} [{default}]: ").strip()
        if text == "":
            return default
        if text.isdigit() and low <= int(text) <= high:
            return int(text)
        print(f"  Please enter a number from {low} to {high}.")


def ask_choice(prompt, options, default):
    """Ask the user to pick one of the numbered options."""
    print(prompt)
    for i, name in enumerate(options, start=1):
        print(f"  {i}. {name}")
    return options[ask_int("Your choice", default, 1, len(options)) - 1]


def ask_yes_no(prompt, default="y"):
    text = input(f"{prompt} (y/n) [{default}]: ").strip().lower()
    return (text or default).startswith("y")


# =============================================================== 2. BUILDING
def build_building(num_exits):
    """
    A 12 x 24 building: top corridor, middle hall and bottom corridor, separated by
    two walls with doorways. Exits are spread along the top and bottom walls.
    Returns (graph, exit_cells).  Node = (row, col).
    """
    grid = [["#"] * COLS for _ in range(ROWS)]
    for r in range(1, ROWS - 1):
        for c in range(1, COLS - 1):
            grid[r][c] = "."
    for c in range(2, COLS - 2):
        if c not in (4, 11, 19):
            grid[4][c] = "#"
        if c not in (7, 14, 20):
            grid[8][c] = "#"

    exits = []
    for i in range(num_exits):
        col = 1 + round((i // 2 + 0.5) * (COLS - 2) / ((num_exits + 1) // 2))
        col = min(max(col, 1), COLS - 2)
        row = 0 if i % 2 == 0 else ROWS - 1
        grid[row][col] = "E"
        exits.append((row, col))

    G = nx.Graph()
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] != "#":
                G.add_node((r, c))
    for (r, c) in list(G.nodes):
        for dr, dc in ((1, 0), (0, 1)):
            if (r + dr, c + dc) in G:
                G.add_edge((r, c), (r + dr, c + dc))
    return G, exits


def place_groups(G, exits, num_groups, fire_start, seed=1):
    """Random starting cells that can reach an exit."""
    rng = random.Random(seed)
    cells = sorted(n for n in G if n not in exits and n != fire_start
                   and any(nx.has_path(G, n, e) for e in exits))
    return rng.sample(cells, num_groups)


def draw_map(G, exits, starts, fire_start):
    """Print the building as text."""
    print("\nBuilding map  (# wall, E exit, digit = group number, F = fire start)")
    for r in range(ROWS):
        line = ""
        for c in range(COLS):
            if (r, c) not in G:
                line += "#"
            elif (r, c) in exits:
                line += "E"
            elif (r, c) in starts:
                line += str(starts.index((r, c)) + 1)
            elif (r, c) == fire_start:
                line += "F"
            else:
                line += "."
        print("  " + line)


# =============================================================== 3. HAZARDS
def make_quake_events(G, exits, starts):
    """Debris falls at steps 2, 4, 6, but every group keeps at least one way out."""
    candidates = sorted(n for n in G if n not in exits and n not in starts)
    seed = 0
    while True:
        rng = random.Random(seed)
        events = {t: set(rng.sample(candidates, 4)) for t in (2, 4, 6)}
        blocked = set().union(*events.values())
        H = G.subgraph(n for n in G if n not in blocked)
        if all(any(nx.has_path(H, s, e) for e in exits) for s in starts):
            return events
        seed += 1


def initial_hazard(scenario, fire_start):
    return {fire_start} if scenario == "Fire" else set()


def update_hazard(scenario, hazard, t, G, quake_events, occupied):
    """Fire spreads every 3 steps / flood rises 1 row every 3 steps / debris at steps 2, 4, 6."""
    if scenario == "Fire":
        if t % 3 == 0:
            new = set(hazard)
            for (r, c) in hazard:
                for dr, dc in DIRS:
                    if (r + dr, c + dc) in G:
                        new.add((r + dr, c + dc))
            return new
        return hazard
    if scenario == "Flood":
        return {n for n in G if n[0] >= ROWS - t // 3}
    return hazard | (quake_events.get(t, set()) - occupied)


def hazard_penalty(node, hazard):
    """Extra cost for walking next to a dangerous cell."""
    r, c = node
    return 5 if any((r + dr, c + dc) in hazard for dr, dc in DIRS) else 0


def hazard_at(scenario, step, G, fire_start, quake_events):
    """The hazard as it looks after 'step' time steps (nobody standing in the way)."""
    hazard = initial_hazard(scenario, fire_start)
    for t in range(1, step + 1):
        hazard = update_hazard(scenario, hazard, t, G, quake_events, set())
    return hazard


# =============================================================== 4. SEARCH ALGORITHMS
def heuristic(node, goals):
    """Manhattan distance to the nearest goal (never overestimates = admissible)."""
    return min(abs(node[0] - g[0]) + abs(node[1] - g[1]) for g in goals)


def rebuild(came_from, node):
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    return path[::-1]


def search(G, start, goals, hazard, algo="A*"):
    """
    algo = 'A*' (f = g + h) | 'Dijkstra' (f = g) | 'BFS' (fewest steps) | 'Greedy' (f = h)
    Returns (path, expanded_nodes); path is None when no route exists.
    """
    goals = set(goals)
    expanded, came_from = 0, {}

    if algo == "BFS":
        frontier, seen = deque([start]), {start}
        while frontier:
            node = frontier.popleft()
            expanded += 1
            if node in goals:
                return rebuild(came_from, node), expanded
            for nb in G.neighbors(node):
                if nb not in seen and nb not in hazard:
                    seen.add(nb)
                    came_from[nb] = node
                    frontier.append(nb)
        return None, expanded

    def priority(g, node):
        if algo == "A*":
            return g + heuristic(node, goals)
        if algo == "Dijkstra":
            return g
        return heuristic(node, goals)                      # Greedy

    heap, g_cost, closed = [(priority(0, start), 0, start)], {start: 0}, set()
    while heap:
        _, g, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node in goals:
            return rebuild(came_from, node), expanded
        for nb in G.neighbors(node):
            if nb in hazard or nb in closed:
                continue
            new_g = g + 1 + hazard_penalty(nb, hazard)
            if algo == "Greedy":
                if nb not in g_cost:
                    g_cost[nb], came_from[nb] = new_g, node
                    heapq.heappush(heap, (priority(new_g, nb), new_g, nb))
            elif new_g < g_cost.get(nb, float("inf")):
                g_cost[nb], came_from[nb] = new_g, node
                heapq.heappush(heap, (priority(new_g, nb), new_g, nb))
    return None, expanded


def path_cost(path, hazard):
    """True cost of a path (same cost model for every algorithm)."""
    return sum(1 + hazard_penalty(n, hazard) for n in path[1:])


def choose_exit(G, group, groups, exits, hazard, balanced, capacity):
    """Best exit for a group: path cost (+ expected waiting time when balanced)."""
    best_exit, best_path, best_score, total = None, None, float("inf"), 0
    for ex in exits:
        path, exp = search(G, group["pos"], [ex], hazard, "A*")
        total += exp
        if path is None:
            continue
        score = path_cost(path, hazard)
        if balanced:
            load = sum(o["size"] for o in groups if o is not group and o["target"] == ex
                       and o["status"] in ("moving", "waiting"))
            score += load / capacity
        if score < best_score:
            best_exit, best_path, best_score = ex, path, score
    return best_exit, best_path, total


# =============================================================== 5. SIMULATION
def snapshot(t, hazard, groups):
    return {"t": t, "hazard": set(hazard),
            "groups": [{"pos": g["pos"], "status": g["status"], "remaining": g["remaining"],
                        "path": list(g["path"] or [])} for g in groups]}


def simulate(G, exits, starts, sizes, capacity, scenario, balanced, fire_start,
             quake_events, verbose=False, record=False):
    """Run the whole evacuation. Returns (result_dict, frames)."""
    hazard = initial_hazard(scenario, fire_start)
    expanded_total = 0
    groups = [{"name": f"Group {i + 1}", "pos": s, "size": sizes[i], "remaining": sizes[i],
               "target": None, "path": None, "status": "moving", "time": None}
              for i, s in enumerate(starts)]

    for g in groups:                                           # first exit assignment
        ex, path, exp = choose_exit(G, g, groups, exits, hazard, balanced, capacity)
        expanded_total += exp
        if ex is None:
            g["status"] = "trapped"
        else:
            g["target"], g["path"] = ex, path
            if verbose:
                print(f"  {g['name']} ({g['size']} people) -> exit {ex}, distance {len(path) - 1}")

    frames = [snapshot(0, hazard, groups)] if record else []
    queues = {ex: [] for ex in exits}
    t = 0
    while any(g["status"] in ("moving", "waiting") for g in groups) and t < 300:
        t += 1
        for g in groups:                                       # a) walk one step
            if g["status"] == "moving":
                g["path"] = g["path"][1:]
                g["pos"] = g["path"][0]
                if g["pos"] == g["target"]:
                    g["status"] = "waiting"
                    queues[g["target"]].append(g)
        for ex, queue in queues.items():                       # b) exits let people out
            free = capacity
            for g in list(queue):
                served = min(free, g["remaining"])
                g["remaining"] -= served
                free -= served
                if g["remaining"] == 0:
                    g["status"], g["time"] = "escaped", t
                    queue.remove(g)
                    if verbose:
                        print(f"  t={t}: {g['name']} is out through exit {ex}")
                if free == 0:
                    break
        occupied = {g["pos"] for g in groups if g["status"] in ("moving", "waiting")}
        hazard = update_hazard(scenario, hazard, t, G, quake_events, occupied)   # c) hazard
        for g in groups:                                       # d) check and re-plan
            if g["status"] == "waiting" and g["pos"] in hazard:
                g["status"] = "trapped"
                queues[g["target"]].remove(g)
                if verbose:
                    print(f"  t={t}: exit {g['target']} is blocked, {g['name']} trapped")
            elif g["status"] == "moving":
                if g["pos"] in hazard:
                    g["status"] = "trapped"
                    if verbose:
                        print(f"  t={t}: {g['name']} was caught by the hazard")
                elif g["target"] in hazard or any(p in hazard for p in g["path"][1:]):
                    if verbose:
                        print(f"  t={t}: {g['name']} route blocked, planning again")
                    ex, path, exp = choose_exit(G, g, groups, exits, hazard, balanced, capacity)
                    expanded_total += exp
                    if ex is None:
                        g["status"] = "trapped"
                        if verbose:
                            print(f"  t={t}: {g['name']} has no safe route left")
                    else:
                        g["target"], g["path"] = ex, path
        if record:
            frames.append(snapshot(t, hazard, groups))

    times = [g["time"] for g in groups if g["status"] == "escaped"]
    result = {"last_out": max(times) if times else None,
              "out": sum(g["size"] - g["remaining"] for g in groups),
              "trapped": sum(1 for g in groups if g["status"] == "trapped"),
              "nodes": expanded_total}
    return result, frames


# =============================================================== 6. COMPARISONS
def compare_strategies(G, exits, starts, sizes, capacity, scenario, fire_start, quake_events, people):
    print("\n=== COMPARISON 1: exit strategies (same building, same disaster) ===")
    print(f"{'Strategy':<16}{'Evacuated':>12}{'Last out (steps)':>20}{'Trapped groups':>16}")
    for name, flag in (("Nearest exit", False), ("Balanced exits", True)):
        r, _ = simulate(G, exits, starts, sizes, capacity, scenario, flag, fire_start, quake_events)
        last = r["last_out"] if r["last_out"] is not None else "-"
        print(f"{name:<16}{str(r['out']) + '/' + str(people):>12}{str(last):>20}{r['trapped']:>16}")


def compare_algorithms(G, exits, starts, scenario, fire_start, quake_events):
    """Every algorithm finds a path for every group, 6 steps after the disaster started."""
    hazard = hazard_at(scenario, 6, G, fire_start, quake_events)
    stats = {a: {"extra": [], "optimal": 0, "nodes": [], "ms": []} for a in ALGORITHMS}
    used = 0
    for s in starts:
        ref, _ = search(G, s, exits, hazard, "Dijkstra")          # Dijkstra = best possible cost
        if ref is None:
            continue
        used += 1
        best = path_cost(ref, hazard)
        for algo in ALGORITHMS:
            t0 = time.perf_counter()
            path, expanded = search(G, s, exits, hazard, algo)
            ms = (time.perf_counter() - t0) * 1000
            cost = path_cost(path, hazard)
            stats[algo]["extra"].append(cost - best)
            stats[algo]["optimal"] += cost == best
            stats[algo]["nodes"].append(expanded)
            stats[algo]["ms"].append(ms)

    print(f"\n=== COMPARISON 2: search algorithms ({used} groups, {scenario.lower()} after 6 steps) ===")
    if not used:
        print("  No group had a route at that time, nothing to compare.")
        return
    print(f"{'Algorithm':<10}{'Avg extra cost':>16}{'Optimal paths':>15}{'Avg nodes':>11}{'Avg time (ms)':>15}")
    for algo in ALGORITHMS:
        s = stats[algo]
        print(f"{algo:<10}{sum(s['extra']) / used:>16.2f}{str(s['optimal']) + '/' + str(used):>15}"
              f"{sum(s['nodes']) / used:>11.1f}{sum(s['ms']) / used:>15.3f}")
    print("  (extra cost 0 = best possible path; fewer nodes = less searching)")


# =============================================================== 7. ANIMATION
def make_animation(G, exits, frames, sizes, scenario, strategy_name):
    total = sum(sizes)
    base = [[1 if (r, c) not in G else 2 if (r, c) in exits else 0 for c in range(COLS)]
            for r in range(ROWS)]
    cmap = colors.ListedColormap(["#f1f3f5", "#343a40", "#51cf66"])
    fig, ax = plt.subplots(figsize=(12, 6))

    def draw(i):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        f = frames[i]
        ax.imshow(base, cmap=cmap, vmin=0, vmax=2)
        for (r, c) in f["hazard"]:
            if base[r][c] != 1:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color=HAZARD_COLOR[scenario], alpha=0.75))
        shared = {}
        for k, g in enumerate(f["groups"]):
            if g["status"] in ("moving", "waiting"):
                shared.setdefault(g["pos"], []).append(k)
        for k, g in enumerate(f["groups"]):
            col = GROUP_COLORS[k % len(GROUP_COLORS)]
            if g["status"] == "escaped":
                continue
            if g["status"] == "trapped":
                ax.plot(g["pos"][1], g["pos"][0], "x", color="red", markersize=16, mew=4)
                continue
            if len(g["path"]) > 1:
                ax.plot([p[1] for p in g["path"]], [p[0] for p in g["path"]],
                        "-", color=col, alpha=0.35, lw=2)
            mates = shared[g["pos"]]
            x = g["pos"][1] + (mates.index(k) - (len(mates) - 1) / 2) * 0.5
            ax.plot(x, g["pos"][0], "o", color=col, markersize=20 if len(mates) == 1 else 14,
                    markeredgecolor="white", mew=2)
            ax.text(x, g["pos"][0], str(g["remaining"]), color="white", ha="center", va="center",
                    fontsize=8, fontweight="bold")
        out = total - sum(g["remaining"] for g in f["groups"])
        trapped = sum(1 for g in f["groups"] if g["status"] == "trapped")
        ax.set_title(f"{scenario} - {strategy_name} | step {f['t']} | evacuated {out}/{total}"
                     + (f" | trapped groups: {trapped}" if trapped else ""), fontsize=12)

    return fig, animation.FuncAnimation(fig, draw, frames=len(frames), interval=400, repeat=False)


def show_animation(G, exits, frames, sizes, scenario, strategy_name):
    if len(frames) > 150:
        frames = frames[:150]
    fig, anim = make_animation(G, exits, frames, sizes, scenario, strategy_name)
    filename = f"evacuation_{scenario.lower()}.gif"
    anim.save(filename, writer=animation.PillowWriter(fps=2))
    print(f"\nAnimation saved as {filename}")
    if IN_NOTEBOOK:
        from IPython.display import HTML, display
        display(HTML(anim.to_jshtml()))
        plt.close(fig)
    elif matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


# =============================================================== 8. MAIN PROGRAM
def main():
    print("=== Disaster Evacuation Planning using Informed Search ===")
    while True:
        print()
        people = ask_int("How many people are in the building?", 100, 1, 2000)
        num_exits = ask_int("How many exits? (1-6)", 3, 1, 6)
        num_groups = ask_int("How many groups should the people be split into? (1-9)", 4, 1, 9)
        capacity = ask_int("How many people can ONE exit let out per step?", 10, 1, 200)
        scenario = ask_choice("\nWhich disaster?", ["Fire", "Flood", "Earthquake"], 1)
        strategy = ask_choice("\nWhich exit strategy?", ["Nearest exit", "Balanced exits"], 2)

        num_groups = min(num_groups, people)
        sizes = [people // num_groups + (1 if i < people % num_groups else 0)
                 for i in range(num_groups)]

        G, exits = build_building(num_exits)
        fire_start = (6, 12)
        starts = place_groups(G, exits, num_groups, fire_start)
        quake_events = make_quake_events(G, exits, starts)

        draw_map(G, exits, starts, fire_start)
        print(f"\nPeople: {people} in {num_groups} groups {sizes} | exits: {num_exits} | "
              f"capacity: {capacity} people per step per exit | disaster: {scenario}")

        balanced = strategy == "Balanced exits"
        print(f"\n--- Evacuation log ({strategy}) ---")
        result, frames = simulate(G, exits, starts, sizes, capacity, scenario, balanced,
                                  fire_start, quake_events, verbose=True, record=True)

        print("\n=== RESULT ===")
        print(f"People evacuated : {result['out']} / {people}")
        print(f"Groups trapped   : {result['trapped']}")
        if result["last_out"] is not None:
            print(f"Total evacuation time (last group out): {result['last_out']} steps")
        print(f"A* nodes expanded: {result['nodes']}")

        compare_strategies(G, exits, starts, sizes, capacity, scenario, fire_start,
                           quake_events, people)
        compare_algorithms(G, exits, starts, scenario, fire_start, quake_events)

        if ask_yes_no("\nMake the animation?", "y"):
            show_animation(G, exits, frames, sizes, scenario, strategy)

        if not ask_yes_no("\nRun again with different numbers?", "n"):
            print("\nThank you!")
            break


if __name__ == "__main__":
    main()
