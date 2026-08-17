"""TensorBoard 로그가 어디까지 기록됐는지 확인."""
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(sys.argv[1] if len(sys.argv) > 1 else "ai/logs")
for run in sorted(root.iterdir()):
    if not run.is_dir():
        continue
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0}); acc.Reload()
    tags = acc.Tags()["scalars"]
    if not tags:
        print(f"{run.name:<40} (스칼라 없음)"); continue
    tag = "rollout/ep_score_mean" if "rollout/ep_score_mean" in tags else tags[0]
    pts = acc.Scalars(tag)
    print(f"{run.name:<40} 마지막 step {pts[-1].step:>10,}  기록 {len(pts):>5}개  "
          f"({tag} 최종 {pts[-1].value:.2f})")
