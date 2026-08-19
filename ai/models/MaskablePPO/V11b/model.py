"""MaskablePPO V11b 신경망 — **V8c/V9c 와 완전히 동일하다. 한 줄도 바꾸지 않았다.**

V11b 가 검증하려는 것은 신경망이 아니라 **학습 신호**다(env.py 참고: 보상을
같은 판 안에서의 상대 점수로 바꿨다). 신경망을 같이 흔들면 무엇이 효과를 냈는지
알 수 없으므로, 지금까지 나온 PPO 계열 구성 중 가장 근거가 확실한 것을 그대로 쓴다.

    전용 critic 트렁크 (V8c vs V8b)   +5.38 ± 1.28   <- V8 에서 가장 큰 승리

아래는 그 V8c 설계의 원래 설명이다.

---

V8c 는 학습된 V 로 빔 탐색을 한다(당시 env.py 의 SEARCH_*). 즉 최종 점수가 정책
로짓보다 **가치 추정의 정확도**에 걸려 있다. 그래서 가치망만 키우고(384x2),
남은 사과 수를 스칼라로 직접 넣어 준다. 남은 사과 수는 남은 점수의 상한이라
가치 예측의 뼈대가 되는데, 풀링된 임베딩에서 이걸 복원하게 두는 것은 낭비다.

(V11b 는 탐색을 쓰지 않지만 이 가치망은 여전히 중요하다. γ=1, λ=1 이라
A_t = R - V(s_t) 인데, 여기서 R 은 에피소드 하나당 값 하나뿐이다. 에피소드 안에서
'어느 수가 좋았나'를 갈라 주는 것은 전적으로 V(s_t) 의 몫이다.)

반대로 트렁크는 V6.0(96ch x 6블록)보다 **줄였다**(64ch x 4블록). V6.0 로그에서
explained_variance 가 0.975 까지 갔으니 표현력은 남아돌았고, 정작 부족했던 건
처리량이다(55 fps, 100만 step 에 5시간).

행동 헤드는 V6.0 그대로다. 두 꼭짓점 임베딩의 내적만으로는 "이 사각형 안에
무엇이 들어있는가"를 표현할 수 없어서, 내부 요약과 기하 사전지식을 더한다.

그래서 로짓을 네 항의 합으로 만든다.

    logit(a) = <tl, br>            꼭짓점 상호작용   — 위치 관계
             + region(a)           내부 요약        — 무엇을 부수는가 (학습됨)
             + bias[shape, count]  기하 x 사과 수   — 판과 무관한 사전지식

내부 요약과 사과 수는 둘 다 **누적합의 네 모서리 뺄셈**으로 구한다. 행동이
7533개여도 gather 네 번이면 전부 계산되므로 비용이 사실상 없다.

가치망은 셀 임베딩을 평균/최대 풀링해서 받는다. 셀을 통째로 편 10368차원을
Linear 에 꽂던 V5.0 방식은 파라미터가 260만 개인 데다 위치마다 따로 배워야 해서
일반화가 나빴다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ai.envs.action_sets import get_all_action
from game.action import Action

WIDTH = 64          # 잔차 블록 채널 수
N_BLOCKS = 4        # 잔차 블록 개수 (수용영역 9x9 + 전역 브랜치)
EMBED_DIM = 48      # 셀별 임베딩 차원
HEAD_DIM = 32       # 꼭짓점 상호작용 차원
VALUE_HIDDEN = 384  # 탐색이 V 에 의존하므로 V6.0(256)보다 키웠다
N_GROUPS = 8        # GroupNorm — 배치 통계에 의존하지 않아 RL 에 안전하다
MAX_APPLE_COUNT = 10  # 합이 10이고 숫자가 1 이상이므로 한 수가 먹는 사과는 최대 10개


def prefix_sum(plane: torch.Tensor) -> torch.Tensor:
    """(N, R, C) -> (N, R+1, C+1) 0 으로 패딩된 2차원 누적합."""
    padded = F.pad(plane, (1, 0, 1, 0))
    return padded.cumsum(dim=-2).cumsum(dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(N_GROUPS, width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(N_GROUPS, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class GridEncoder(BaseFeaturesExtractor):
    """관측 -> [셀별 임베딩 | 사과 점유 누적합] 을 이어 붙인 벡터.

    누적합을 특징에 같이 실어 보내는 이유: SB3 는 행동 헤드에 관측을 넘겨주지
    않는다. 헤드가 직사각형 안의 사과 수를 알려면 이 경로밖에 없다.
    """

    def __init__(self, observation_space, width: int = WIDTH,
                 n_blocks: int = N_BLOCKS, embed_dim: int = EMBED_DIM):
        n_channels, rows, cols = observation_space.shape  # type: ignore
        self.cell_dim = embed_dim * rows * cols
        super().__init__(observation_space, features_dim=self.cell_dim + (rows + 1) * (cols + 1))

        self.rows, self.cols, self.embed_dim = rows, cols, embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(N_GROUPS, width), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        # 전역 브랜치: 9x18 판은 잔차 블록의 수용영역보다 넓다
        self.global_fc = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU())
        self.mix = nn.Sequential(nn.Conv2d(2 * width, embed_dim, kernel_size=1), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(observations))                    # (N, W, R, C)

        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        g = self.global_fc(pooled)[:, :, None, None].expand(-1, -1, self.rows, self.cols)
        cells = self.mix(torch.cat([h, g], dim=1))                  # (N, E, R, C)

        occupied = 1.0 - observations[:, 0]                         # 0번 평면 = 빈칸
        return torch.cat([cells.flatten(1), prefix_sum(occupied).flatten(1)], dim=1)


class RectangleHead(nn.Module):
    """셀 임베딩과 점유 누적합에서 7533개 직사각형의 로짓을 한 번에 만든다."""

    def __init__(self, rows: int, cols: int, embed_dim: int, cell_dim: int,
                 actions: list[Action], head_dim: int = HEAD_DIM):
        super().__init__()
        self.rows, self.cols, self.embed_dim, self.cell_dim = rows, cols, embed_dim, cell_dim

        self.proj_tl = nn.Linear(embed_dim, head_dim)
        self.proj_br = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5
        self.region = nn.Conv2d(embed_dim, 1, kernel_size=1)  # 내부 요약용 평면

        r1 = torch.tensor([a.top_left[0] for a in actions], dtype=torch.long)
        c1 = torch.tensor([a.top_left[1] for a in actions], dtype=torch.long)
        r2 = torch.tensor([a.bottom_right[0] for a in actions], dtype=torch.long)
        c2 = torch.tensor([a.bottom_right[1] for a in actions], dtype=torch.long)

        self.register_buffer("tl_idx", r1 * cols + c1)
        self.register_buffer("br_idx", r2 * cols + c2)
        # 누적합 뺄셈용 모서리 좌표 (hi 쪽은 반열린 구간이라 +1)
        self.register_buffer("r_lo", r1)
        self.register_buffer("c_lo", c1)
        self.register_buffer("r_hi", r2 + 1)
        self.register_buffer("c_hi", c2 + 1)

        # (높이, 너비) x (사과 수) 사전지식 표. "넓기만 하고 안 먹는 수"를 여기서 배운다.
        self.n_counts = MAX_APPLE_COUNT + 1
        self.register_buffer("shape_idx", (r2 - r1) * cols + (c2 - c1))
        self.shape_count_bias = nn.Parameter(torch.zeros(rows * cols, self.n_counts))

        # 초기 로짓을 0 근처로 -> 시작 정책이 (마스크 안에서) 균등에 가깝게
        for layer in (self.proj_tl, self.proj_br):
            nn.init.orthogonal_(layer.weight, gain=0.3)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.region.weight)
        nn.init.zeros_(self.region.bias)

    def _rect_sum(self, prefix: torch.Tensor) -> torch.Tensor:
        """(N, R+1, C+1) 누적합 -> 행동별 직사각형 합 (N, A)."""
        return (prefix[:, self.r_hi, self.c_hi] - prefix[:, self.r_lo, self.c_hi]
                - prefix[:, self.r_hi, self.c_lo] + prefix[:, self.r_lo, self.c_lo])

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        n = latent_pi.shape[0]
        cells = latent_pi[:, :self.cell_dim].view(n, self.embed_dim, self.rows, self.cols)
        occ_prefix = latent_pi[:, self.cell_dim:].view(n, self.rows + 1, self.cols + 1)

        # 1) 꼭짓점 상호작용
        flat = cells.flatten(2).transpose(1, 2)                       # (N, R*C, E)
        tl = self.proj_tl(flat)
        br = self.proj_br(flat)
        pair = torch.bmm(tl, br.transpose(1, 2)) * self.scale         # (N, R*C, R*C)
        logits = pair[:, self.tl_idx, self.br_idx]                    # (N, A)

        # 2) 내부 요약 — 이 수가 무엇을 부수는가
        logits = logits + self._rect_sum(prefix_sum(self.region(cells).squeeze(1)))

        # 3) 기하 x 사과 수 사전지식
        count = self._rect_sum(occ_prefix).round().long().clamp_(0, MAX_APPLE_COUNT)
        bias = self.shape_count_bias.view(-1)[self.shape_idx * self.n_counts + count]
        return logits + bias


class PooledCriticExtractor(nn.Module):
    """정책에는 특징을 그대로, 가치망에는 풀링된 셀 임베딩 + 남은 사과 수를 넘긴다."""

    def __init__(self, features_dim: int, cell_dim: int, embed_dim: int,
                 n_cells: int, hidden: int = VALUE_HIDDEN):
        super().__init__()
        self.cell_dim, self.embed_dim, self.n_cells = cell_dim, embed_dim, n_cells
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = hidden
        self.value_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        cells = features[:, :self.cell_dim].view(features.shape[0], self.embed_dim, -1)
        # 점유 누적합의 마지막 칸 = 남은 사과 총수 = 남은 점수의 상한
        remaining = features[:, -1:] / self.n_cells
        pooled = torch.cat([cells.mean(dim=2), cells.amax(dim=2), remaining], dim=1)
        return self.value_mlp(pooled)


class RectangleMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy 에서 mlp_extractor 와 action_net 을 갈아끼운다.

    사용 조건: features_extractor 는 GridEncoder (rows/cols/embed_dim/cell_dim 필요).
    net_arch 는 쓰이지 않는다 (mlp_extractor 를 직접 만들기 때문).
    """

    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)를 넘겨주세요."
        self._actions = actions
        self._head_dim = head_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        fe = self.features_extractor  # GridEncoder
        self.mlp_extractor = PooledCriticExtractor(
            fe.features_dim, fe.cell_dim, fe.embed_dim, fe.rows * fe.cols,  # type: ignore[arg-type]
        )

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        fe = self.features_extractor
        self.action_net = RectangleHead(
            fe.rows, fe.cols, fe.embed_dim, fe.cell_dim,  # type: ignore[arg-type]
            self._actions, self._head_dim,
        )
        # action_net 을 교체했으므로 옵티마이저 재생성 (새 파라미터 포함, 기존 Linear 제외)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs  # type: ignore
        )


POLICY_CLASS = RectangleMaskablePolicy


def make_policy_kwargs(rows: int, cols: int) -> dict:
    return dict(
        features_extractor_class=GridEncoder,
        features_extractor_kwargs=dict(width=WIDTH, n_blocks=N_BLOCKS, embed_dim=EMBED_DIM),
        net_arch=dict(pi=[], vf=[]),          # 쓰이지 않는다 (_build_mlp_extractor 를 덮어씀)
        normalize_images=False,
        actions=get_all_action(rows, cols),   # RectangleMaskablePolicy 전용 인자
        head_dim=HEAD_DIM,
        # V8c 의 유일한 차이: 가치망에 전용 트렁크를 준다.
        # 지금까지 정책과 가치가 같은 인코더를 공유했는데, 정책 경사가 훨씬 세서
        # 트렁크가 '어떤 수를 둘까'에 맞춰 눌린다. 탐색이 필요로 하는 것은
        # '형제 상태 중 어느 쪽이 나은가' 라는 다른 표현이므로, 갈라 놓고
        # V8b 와 비교하면 그 간섭이 실제 원인이었는지가 나온다. (파라미터는 2배)
        share_features_extractor=False,
    )
