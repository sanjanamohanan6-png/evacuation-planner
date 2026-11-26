"""
Evacuation Planner - Streamlit web application
Uses the functions from evacuation_complete.py (keep both files in the same folder).

Run locally:  pip install -r requirements.txt
              streamlit run app.py
"""
import os
import tempfile
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib import animation, colors

import evacuation_complete as ev

st.set_page_config(page_title="Evacuation Planner", page_icon="🚪", layout="wide")


# ------------------------------------------------------------------ computation
def compute(people, num_exits, num_groups, capacity, scenario, strategy):
    """Run the whole evacuation and every comparison. Returns a dict of results."""
    num_groups = min(num_groups, people)
    sizes = [people // num_groups + (1 if i < people % num_groups else 0)
             for i in range(num_groups)]
    G, exits = ev.build_building(num_exits)
    fire_start = (6, 12)
    starts = ev.place_groups(G, exits, num_groups, fire_start)
    quake = ev.make_quake_events(G, exits, starts)

    balanced = strategy == "Balanced exits"
    result, frames = ev.simulate(G, exits, starts, sizes, capacity, scenario, balanced,
                                 fire_start, quake, record=True)

    # comparison 1: exit strategies
    strat_rows = []
    for name, flag in (("Nearest exit", False), ("Balanced exits", True)):
        r, _ = ev.simulate(G, exits, starts, sizes, capacity, scenario, flag, fire_start, quake)
        strat_rows.append({"Strategy": name, "People evacuated": f"{r['out']} / {people}",
                           "Last group out (steps)": r["last_out"] if r["last_out"] is not None else "-",
                           "Trapped groups": r["trapped"]})

    # comparison 2: search algorithms, 6 steps after the disaster started
    hazard = ev.hazard_at(scenario, 6, G, fire_start, quake)
    stats = {a: {"extra": [], "optimal": 0, "nodes": [], "ms": []} for a in ev.ALGORITHMS}
    used = 0
    for s in starts:
        ref, _ = ev.search(G, s, exits, hazard, "Dijkstra")
        if ref is None:
            continue
        used += 1
        best = ev.path_cost(ref, hazard)
        for algo in ev.ALGORITHMS:
            t0 = time.perf_counter()
            path, expanded = ev.search(G, s, exits, hazard, algo)
            ms = (time.perf_counter() - t0) * 1000
            cost = ev.path_cost(path, hazard)
            stats[algo]["extra"].append(cost - best)
            stats[algo]["optimal"] += cost == best
            stats[algo]["nodes"].append(expanded)
            stats[algo]["ms"].append(ms)
    algo_rows = []
    for algo in ev.ALGORITHMS if used else []:
        s = stats[algo]
        algo_rows.append({"Algorithm": algo,
                          "Avg extra cost": round(sum(s["extra"]) / used, 2),
                          "Optimal paths": f"{s['optimal']} / {used}",
                          "Avg cells examined": round(sum(s["nodes"]) / used, 1),
                          "Avg time (ms)": round(sum(s["ms"]) / used, 3)})

    return {"people": people, "sizes": sizes, "G": G, "exits": exits, "starts": starts,
            "fire_start": fire_start, "scenario": scenario, "strategy": strategy,
            "result": result, "frames": frames[:150], "strat_rows": strat_rows,
            "algo_rows": algo_rows, "used": used}


def building_text(run):
    lines = []
    for r in range(ev.ROWS):
        line = ""
        for c in range(ev.COLS):
            n = (r, c)
            if n not in run["G"]:
                line += "#"
            elif n in run["exits"]:
                line += "E"
            elif n in run["starts"]:
                line += str(run["starts"].index(n) + 1)
            elif n == run["fire_start"]:
                line += "F"
            else:
                line += "."
        lines.append(line)
    return "\n".join(lines)


# ------------------------------------------------------------------ drawing
def draw_frame(run, i):
    """One time step of the evacuation as a picture."""
    G, exits, scenario = run["G"], run["exits"], run["scenario"]
    f = run["frames"][i]
    base = [[1 if (r, c) not in G else 2 if (r, c) in exits else 0 for c in range(ev.COLS)]
            for r in range(ev.ROWS)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(base, cmap=colors.ListedColormap(["#f1f3f5", "#343a40", "#51cf66"]), vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for (r, c) in f["hazard"]:
        if base[r][c] != 1:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       color=ev.HAZARD_COLOR[scenario], alpha=0.75))
    shared = {}
    for k, g in enumerate(f["groups"]):
        if g["status"] in ("moving", "waiting"):
            shared.setdefault(g["pos"], []).append(k)
    for k, g in enumerate(f["groups"]):
        col = ev.GROUP_COLORS[k % len(ev.GROUP_COLORS)]
        if g["status"] == "escaped":
            continue
        if g["status"] == "trapped":
            ax.plot(g["pos"][1], g["pos"][0], "x", color="red", markersize=16, mew=4)
            continue
        if len(g["path"]) > 1:
            ax.plot([p[1] for p in g["path"]], [p[0] for p in g["path"]], "-", color=col, alpha=0.35, lw=2)
        mates = shared[g["pos"]]
        x = g["pos"][1] + (mates.index(k) - (len(mates) - 1) / 2) * 0.5
        ax.plot(x, g["pos"][0], "o", color=col, markersize=20 if len(mates) == 1 else 14,
                markeredgecolor="white", mew=2)
        ax.text(x, g["pos"][0], str(g["remaining"]), color="white", ha="center", va="center",
                fontsize=8, fontweight="bold")
    out = sum(run["sizes"]) - sum(g["remaining"] for g in f["groups"])
    ax.set_title(f"{scenario} | step {f['t']} | evacuated {out}/{sum(run['sizes'])}", fontsize=12)
    return fig


def make_gif(run):
    """Animated GIF of the whole evacuation, returned as bytes."""
    fig, anim = ev.make_animation(run["G"], run["exits"], run["frames"], run["sizes"],
                                  run["scenario"], run["strategy"])
    path = os.path.join(tempfile.gettempdir(), "evacuation_streamlit.gif")
    anim.save(path, writer=animation.PillowWriter(fps=2))
    plt.close(fig)
    with open(path, "rb") as fh:
        return fh.read()


# ------------------------------------------------------------------ pages (step by step)
STEPS = ["Welcome", "People", "Building", "Disaster", "Strategy", "Review", "Results"]
DEFAULTS = {"people": 120, "groups": 5, "exits": 3, "capacity": 10,
            "scenario": "Fire", "strategy": "Balanced exits"}
DISASTERS = {"Fire": "Starts in the middle of the building and spreads to the next squares every 3 steps.",
             "Flood": "Water rises from the bottom, one row every 3 steps, so the lower exits close first.",
             "Earthquake": "Debris suddenly blocks some squares at steps 2, 4 and 6."}
STRATEGIES = {"Balanced exits": "Each group compares the path length AND the waiting line at each exit.",
              "Nearest exit": "Each group simply goes to the exit with the shortest path."}

st.session_state.setdefault("page", 0)
cfg = st.session_state.setdefault("cfg", dict(DEFAULTS))
page = st.session_state["page"]


def go(n):
    st.session_state["page"] = n


def run_now():
    st.session_state["page"] = 6
    st.session_state["need_run"] = True


def restart():
    st.session_state["cfg"] = dict(DEFAULTS)
    st.session_state["page"] = 0
    st.session_state.pop("run", None)


def nav(back, nxt, label="Next"):
    c1, c2, _ = st.columns([1, 1, 4])
    if back is not None:
        c1.button("← Back", on_click=go, args=(back,), key=f"back{page}")
    if nxt is not None:
        c2.button(f"{label} →", type="primary", on_click=go, args=(nxt,), key=f"next{page}")


# ---- header and sidebar
st.title("🚪 Evacuation Planner")
with st.sidebar:
    st.header("Steps")
    for i, name in enumerate(STEPS):
        st.markdown(("**👉 " + name + "**") if i == page else (("✅ " if i < page else "⚪ ") + name))
    st.divider()
    st.caption("A* route planning with exit capacity, spreading hazards and dynamic re-planning.")
if 0 < page < 6:
    st.progress(page / 5, text=f"Question {page} of 5")

# ---- 0. Welcome
if page == 0:
    st.header("Welcome")
    with st.container(border=True):
        st.write("This application plans how to evacuate a building during a **fire, flood or earthquake**.")
        st.write("You answer a few simple questions. The program then finds the best route for each "
                 "group of people with the **A\\* search algorithm**, chooses the exits, and plans "
                 "again when a route gets blocked.")
        st.markdown("1. How many people are in the building?\n2. How many exits does it have?\n"
                    "3. Which disaster happens?\n4. Which exit strategy should be used?")
    st.button("Start →", type="primary", on_click=go, args=(1,))

# ---- 1. People
elif page == 1:
    st.header("How many people are in the building?")
    with st.container(border=True):
        cfg["people"] = st.number_input("Number of people", 1, 2000, int(cfg["people"]), step=10)
        cfg["groups"] = st.slider("Split them into how many groups?", 1, 9, int(cfg["groups"]))
        st.caption("People move in groups. Each group follows one route at a time.")
    nav(0, 2)

# ---- 2. Building
elif page == 2:
    st.header("How many exits does the building have?")
    with st.container(border=True):
        cfg["exits"] = st.slider("Number of exits", 1, 6, int(cfg["exits"]))
        cfg["capacity"] = st.number_input("People one exit can let out per step", 1, 200, int(cfg["capacity"]))
        G_prev, ex_prev = ev.build_building(int(cfg["exits"]))
        rows = ["".join("E" if (r, c) in ex_prev else "." if (r, c) in G_prev else "#"
                        for c in range(ev.COLS)) for r in range(ev.ROWS)]
        st.code("\n".join(rows), language=None)
        st.caption("Building preview: # wall, . floor, E exit")
    nav(1, 3)

# ---- 3. Disaster
elif page == 3:
    st.header("Which disaster happens?")
    with st.container(border=True):
        cfg["scenario"] = st.radio("Disaster", list(DISASTERS), index=list(DISASTERS).index(cfg["scenario"]))
        st.info(DISASTERS[cfg["scenario"]])
    nav(2, 4)

# ---- 4. Strategy
elif page == 4:
    st.header("Which exit strategy should be used?")
    with st.container(border=True):
        cfg["strategy"] = st.radio("Strategy", list(STRATEGIES),
                                   index=list(STRATEGIES).index(cfg["strategy"]))
        st.info(STRATEGIES[cfg["strategy"]])
    nav(3, 5)

# ---- 5. Review
elif page == 5:
    st.header("Review your answers")
    with st.container(border=True):
        st.dataframe(pd.DataFrame({
            "Question": ["People", "Groups", "Exits", "People per exit per step", "Disaster", "Exit strategy"],
            "Your answer": [str(x) for x in (cfg["people"], cfg["groups"], cfg["exits"], cfg["capacity"],
                                             cfg["scenario"], cfg["strategy"])]}), hide_index=True, width="stretch")
    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("← Back", on_click=go, args=(4,), key="back5")
    c2.button("Run evacuation", type="primary", on_click=run_now, key="run5")

# ---- 6. Results
else:
    if st.session_state.pop("need_run", False) or "run" not in st.session_state:
        with st.spinner("Planning the evacuation..."):
            st.session_state["run"] = compute(int(cfg["people"]), int(cfg["exits"]), int(cfg["groups"]),
                                              int(cfg["capacity"]), cfg["scenario"], cfg["strategy"])
            st.session_state["gif"] = make_gif(st.session_state["run"])
            st.session_state["step"] = 0
    run = st.session_state["run"]
    res = run["result"]

    st.header("Results")
    st.caption(f"{run['scenario']} | {run['strategy']} | {run['people']} people in groups {run['sizes']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People evacuated", f"{res['out']} / {run['people']}")
    c2.metric("Last group out (steps)", res["last_out"] if res["last_out"] is not None else "-")
    c3.metric("Groups trapped", res["trapped"])
    c4.metric("A* cells examined", res["nodes"])

    tab_gif, tab_step, tab_cmp, tab_map = st.tabs(["Animation", "Step by step", "Comparisons", "Building map"])
    with tab_gif:
        st.image(st.session_state["gif"], width="stretch")
        st.caption("Circles are groups (number = people still inside). Thin lines are planned routes. Red X = trapped.")
    with tab_step:
        step = st.slider("Time step", 0, len(run["frames"]) - 1, key="step")
        fig = draw_frame(run, step)
        st.pyplot(fig)
        plt.close(fig)
    with tab_cmp:
        st.subheader("Exit strategies")
        st.dataframe(pd.DataFrame(run["strat_rows"]), hide_index=True, width="stretch")
        st.subheader("Search algorithms")
        if run["algo_rows"]:
            st.dataframe(pd.DataFrame(run["algo_rows"]), hide_index=True, width="stretch")
            st.caption(f"{run['used']} groups, {run['scenario'].lower()} after 6 steps. "
                       "Extra cost 0 = best possible path; fewer cells examined = less searching.")
        else:
            st.info("No group had a route at that time, so there is nothing to compare.")
    with tab_map:
        st.code(building_text(run), language=None)
        st.caption("# wall, E exit, digit = group number, F = fire start")

    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Change answers", on_click=go, args=(1,), key="change6")
    b2.button("Start again", on_click=restart, key="restart6")
