#!/usr/bin/env python3
"""
run_comparison.py가 만든 비교 JSON을 읽어서
버전별(avg_score, std_score, clear_rate, avg_turn 등) 통계 대시보드 HTML을 생성한다.

사용법:
    python generate_dashboard.py <comparison.json> [output.html]

출력:
    단일 HTML 파일 (외부 서버 불필요, 브라우저로 바로 열면 됨)
"""
import json
import sys
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Apple Game 알고리즘 비교 대시보드</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1115;
    --panel: #171a21;
    --border: #262b36;
    --text: #e6e8ec;
    --muted: #8b92a3;
    --accent: #ff5c5c;
    --accent2: #5c9dff;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "Pretendard", "Segoe UI", sans-serif;
    margin: 0;
    padding: 32px;
  }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .meta { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin-bottom: 24px;
  }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
  }
  .panel h2 { font-size: 14px; color: var(--muted); font-weight: 600; margin: 0 0 14px; text-transform: uppercase; letter-spacing: 0.03em; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  tr:hover td { background: #1d212b; }
  .best { color: #6fe08f; font-weight: 600; }
  canvas { max-height: 320px; }
</style>
</head>
<body>
  <h1>🍎 Apple Game 알고리즘 비교 대시보드</h1>
  <div class="meta" id="meta"></div>

  <div class="grid">
    <div class="panel">
      <h2>평균 점수 (avg_score)</h2>
      <canvas id="avgScoreChart"></canvas>
    </div>
    <div class="panel">
      <h2>안정성: 평균 vs 표준편차</h2>
      <canvas id="stabilityChart"></canvas>
    </div>
    <div class="panel">
      <h2>올클리어 비율 (clear_rate)</h2>
      <canvas id="clearRateChart"></canvas>
    </div>
    <div class="panel">
      <h2>평균 턴 수 (avg_turn)</h2>
      <canvas id="avgTurnChart"></canvas>
    </div>
  </div>

  <div class="panel">
    <h2>상세 수치</h2>
    <table id="table"></table>
  </div>

<script>
const DATA = __DATA__;

const runs = DATA.runs;
const labels = runs.map(r => r.name);
const colors = ["#ff5c5c", "#5c9dff", "#6fe08f", "#f7c948", "#c084fc", "#4fd1c5", "#f97316", "#a3a3a3"];

document.getElementById("meta").textContent =
  `생성 시각: ${DATA.generated_at} | 버전당 게임 수: ${DATA.n_games}`;

function bestIndex(key, higherIsBetter = true) {
  let best = 0;
  for (let i = 1; i < runs.length; i++) {
    const a = runs[i].summary[key], b = runs[best].summary[key];
    if (higherIsBetter ? a > b : a < b) best = i;
  }
  return best;
}

const baseOptions = {
  responsive: true,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: "#8b92a3" }, grid: { color: "#262b36" } },
    y: { ticks: { color: "#8b92a3" }, grid: { color: "#262b36" } }
  }
};

new Chart(document.getElementById("avgScoreChart"), {
  type: "bar",
  data: {
    labels,
    datasets: [{
      data: runs.map(r => r.summary.avg_score),
      backgroundColor: colors,
    }]
  },
  options: baseOptions
});

new Chart(document.getElementById("clearRateChart"), {
  type: "bar",
  data: {
    labels,
    datasets: [{
      data: runs.map(r => r.summary.clear_rate * 100),
      backgroundColor: colors,
    }]
  },
  options: baseOptions
});

new Chart(document.getElementById("avgTurnChart"), {
  type: "bar",
  data: {
    labels,
    datasets: [{
      data: runs.map(r => r.summary.avg_turn),
      backgroundColor: colors,
    }]
  },
  options: baseOptions
});

new Chart(document.getElementById("stabilityChart"), {
  type: "scatter",
  data: {
    datasets: runs.map((r, i) => ({
      label: r.name,
      data: [{ x: r.summary.avg_score, y: r.summary.std_score }],
      backgroundColor: colors[i % colors.length],
      pointRadius: 8,
    }))
  },
  options: {
    ...baseOptions,
    plugins: {
      legend: { display: true, labels: { color: "#8b92a3" } },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: avg=${ctx.parsed.x.toFixed(1)}, std=${ctx.parsed.y.toFixed(1)}`
        }
      }
    },
    scales: {
      x: { title: { display: true, text: "avg_score (높을수록 좋음)", color: "#8b92a3" }, ticks: { color: "#8b92a3" }, grid: { color: "#262b36" } },
      y: { title: { display: true, text: "std_score (낮을수록 안정적)", color: "#8b92a3" }, ticks: { color: "#8b92a3" }, grid: { color: "#262b36" } }
    }
  }
});

// 상세 테이블
const bestAvg = bestIndex("avg_score", true);
const bestStd = bestIndex("std_score", false);
const bestClear = bestIndex("clear_rate", true);

const cols = [
  ["name", "버전"],
  ["avg_score", "평균 점수"],
  ["std_score", "표준편차"],
  ["max_score", "최고 점수"],
  ["min_score", "최저 점수"],
  ["avg_turn", "평균 턴"],
  ["clear_rate", "올클리어율"],
  ["avg_time", "게임당 시간(s)"],
];

let html = "<tr>" + cols.map(c => `<th>${c[1]}</th>`).join("") + "</tr>";
runs.forEach((r, i) => {
  html += "<tr>";
  cols.forEach(([key]) => {
    let val;
    if (key === "name") {
      val = r.name;
    } else {
      const raw = r.summary[key];
      val = key === "clear_rate" ? (raw * 100).toFixed(1) + "%" : raw.toFixed(2);
      if ((key === "avg_score" && i === bestAvg) ||
          (key === "std_score" && i === bestStd) ||
          (key === "clear_rate" && i === bestClear)) {
        val = `<span class="best">${val}</span>`;
      }
    }
    html += `<td>${val}</td>`;
  });
  html += "</tr>";
});
document.getElementById("table").innerHTML = html;
</script>
</body>
</html>
"""


def build_html(data: dict) -> str:
    """비교 데이터(dict)를 받아 완성된 대시보드 HTML 문자열을 반환한다.
    다른 스크립트(comparison_test.py)에서 매 run 직후 즉시 재생성할 때 재사용."""
    return HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))


def generate_dashboard_file(json_path: Path, out_path: Path | None = None) -> Path:
    """JSON 파일 경로를 받아 대시보드 HTML 파일을 생성하고 그 경로를 반환한다."""
    if out_path is None:
        out_path = json_path.with_suffix(".html")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out_path.write_text(build_html(data), encoding="utf-8")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("사용법: python generate_dashboard.py <comparison.json> [output.html]")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"파일을 찾을 수 없음: {json_path}")
        sys.exit(1)

    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    result_path = generate_dashboard_file(json_path, out_path)

    print(f"대시보드 생성 완료: {result_path}")


if __name__ == "__main__":
    main()