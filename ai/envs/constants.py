STEP_REWARD_TOTAL = 1.0

# step 보상의 합이 이미 (최종점수/전체칸수) * STEP_REWARD_TOTAL 이므로,
# terminal 보상은 같은 양을 한 번 더 주는 것에 불과함 (정책 순위를 전혀 바꾸지 못함).
# 지연 보상만 늘려 value 학습과 credit assignment를 어렵게 하므로 제거.
TERMINAL_REWARD_TOTAL = 0.0

REWARD_ALL_CLEAR_BONUS = 0.1  # 희소하므로 낮게 책정