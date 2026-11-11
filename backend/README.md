# RescueWay
## RL-based Emergency Traffic Signal Management System

### Tech Stack
- Frontend: React + Vite + Tailwind CSS
- Backend: FastAPI + Python
- ML: Tabular Q-Learning (NumPy) + Double DQN (PyTorch, `backend/sumo/`)
- Hardware: Arduino Uno + 12 LEDs
- Communication: PySerial (COM20) + WebSocket

### Hardware Pin Map
| Zone | Red | Yellow | Green |
|------|-----|--------|-------|
| 1    | D2  | D3     | D4    |
| 2    | D5  | D6     | D7    |
| 3    | D8  | D9     | D10   |
| 4    | D11 | D12    | D13   |

### How to Run

**Step 1 — Upload Arduino firmware**
Open arduino/traffic_controller.ino in Arduino IDE
Upload to Arduino Uno on COM20

**Step 2 — Train RL agent (run once)**
cd backend
python train_offline.py --episodes 25000 --seed 42
(held-out pattern for train/test split defaults to `rush_hour`; see `ARCHITECTURE.md`
Part 8.8/8.11 for the multi-seed variance methodology if retraining for research purposes)

**Step 3 — Start backend**
cd backend
uvicorn main:app --reload --port 8000

**Step 4 — Start frontend**
cd frontend
npm run dev

**Step 5 — Open dashboard**
http://localhost:5173

### Training Results
(current as of 2026-07-21 — see `ARCHITECTURE.md` Part 8 for the design-validation-report
implementation pass, and Part 9 for the emergency-accuracy counting fix these numbers
already reflect)

- **Tabular Q-Learning** (canonical seed=42, 25,000 episodes, `rush_hour` held out as a
  test pattern): final avg reward +572.42, emergency accuracy 56.8%, avg ambulance delay
  0.09s, ambulance stop rate 4.5%. Held-out pattern performs within noise of in-distribution
  patterns (-1.8pp accuracy gap) — no evidence of overfitting. **5-seed variance** (seeds
  42/1337/7777/2024/9999): emergency accuracy 56.08% ± 0.76%, reward 570.79 ± 28.93 — a
  seed-stable result (`results/qlearning_training/multi_seed_variance_summary.json`).
- **DQN** (750 episodes, low/medium/high curriculum with replay-buffer carryover across
  stages, `rush_hour` held out): 89.7% / 91.4% / 86.9% emergency accuracy on the three
  trained patterns (no catastrophic forgetting — earliest-trained pattern scores highest,
  the opposite of what forgetting would predict), 69.8% on the held-out `rush_hour` pattern
  (`sumo/training_logs/multi_pattern_20260721_104430.json`). **5-seed variance** (200
  episodes each, single pattern): emergency accuracy 67.7% ± 16.2pp — markedly higher
  seed-sensitivity than Q-Learning at comparable scale, consistent with Henderson et al.
  (2018)'s findings on neural-network RL methods.
- **Flagship benchmark** (`run_comparison.py`, simplified environment, 200 episodes):
  Double DQN + Safety Shield now the top-ranked non-oracle controller (reward +1119.18,
  100.0% emergency accuracy — the shield guarantees the correct zone every tick), ahead of
  Q-Learning (+1097.79, 78.7%) and the new Max-Pressure baseline (+825.08, 23.1%).
  Rule-Based (100.0%/+1308.90) is reported separately as an oracle reference, not ranked
  against learned policies that lack its perfect-information hard-priority shortcut.