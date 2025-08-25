# --- 1. 설치 및 라이브러리 임포트 ---
import os
import random
import time
import asyncio  # 비동기 처리를 위한 라이브러리
import openai
from google import genai
from google.genai import types
import pandas as pd
from dotenv import load_dotenv
import nest_asyncio

load_dotenv()

# --- 헬퍼 함수 ---


def log_message(message, type="info"):
    """타임스탬프와 함께 로그 메시지 형식을 지정하는 헬퍼 함수."""
    timestamp = time.strftime("%H:%M:%S")
    if type == "success":
        return f"[{timestamp}] ✅ 성공: {message}"
    if type == "fail":
        return f"[{timestamp}] ❌ 실패: {message}"
    if type == "best":
        return f"[{timestamp}] ⭐ 최고: {message}"
    return f"[{timestamp}] ℹ️ 정보: {message}"


# --- GEPA 핵심 함수 ---


async def run_openai_rollout_async(
    client: openai.AsyncClient, model_id: str, prompt: str, task: dict
) -> tuple[int, dict]:
    """
    대상 모델(gpt-4o-mini)에 대해 OpenAI API를 비동기적으로 호출합니다.
    """
    try:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task["input"]},
        ]
        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=5,
            temperature=0.4,  # 대회 규정에 맞춰 0.4로 고정
            top_p=0.95,
        )
        output = response.choices[0].message.content.strip()
        eval_result = evaluation_and_feedback_function(output, task)
        return task["id"], eval_result
    except Exception as e:
        # 오류 발생 시 해당 작업은 0점 처리
        return task["id"], {"score": 0.0, "feedback": f"API 호출 오류: {e}"}


def evaluation_and_feedback_function(output, task):
    """
    모델의 출력이 정확히 '0' 또는 '1'인지 채점하고 피드백을 제공합니다. (동기 함수)
    """
    expected_output = task.get("expected_output")
    if output == expected_output:
        return {"score": 1.0, "feedback": "성공"}
    else:
        return {
            "score": 0.0,
            "feedback": f"실패 (예상: {expected_output}, 결과: {output})",
        }


async def evaluate_candidate_async(
    async_openai_client, model_id, prompt, training_data
):
    """
    하나의 프롬프트 후보에 대해 전체 훈련 데이터를 병렬(비동기)로 평가합니다.
    """
    tasks = [
        run_openai_rollout_async(async_openai_client, model_id, prompt, task)
        for task in training_data
    ]
    results = await asyncio.gather(*tasks)

    # 결과를 원래 순서대로 정렬
    sorted_results = sorted(results, key=lambda x: x[0])
    scores = [res[1]["score"] for res in sorted_results]
    total_score = sum(scores)
    avg_score = total_score / len(training_data) if training_data else 0.0

    return scores, avg_score


def reflect_and_propose_new_prompt(
    gemini_client: genai.Client, reflector_model_name, current_prompt, examples
):
    """
    강력한 LLM(Gemini)을 사용하여 '반성적 프롬프트 변형' 단계를 수행합니다. (동기 함수)
    """
    examples_text = "---\n".join(
        f'과제 입력:\n{e["input"]}\n\n생성된 출력: "{e["output"]}"\n피드백: {e["feedback"]}\n\n'
        for e in examples
    )

    reflection_prompt = f"""당신은 전문 프롬프트 엔지니어입니다. 당신의 임무는 이전 시도에서 얻은 피드백을 바탕으로 gpt-4o-mini 모델의 성능을 개선하는 것입니다.

    개선이 필요한 현재 프롬프트는 다음과 같습니다:
    --- 현재 프롬프트 ---
    {current_prompt}
    --------------------

    다음은 이 프롬프트가 몇 가지 과제에서 어떻게 수행되었는지, 그리고 무엇이 잘되었고 잘못되었는지에 대한 피드백 예시입니다:
    --- 예시 및 피드백 ---
    {examples_text}
    -------------------------

    이 분석을 바탕으로, 새롭고 개선된 프롬프트를 작성하는 것이 당신의 임무입니다.
    새로운 프롬프트는 다음 규칙을 반드시 따라야 합니다:
    1. 실패 원인을 직접적으로 해결하고 성공 전략을 반영해야 합니다.
    2. 평가에 길이 점수가 있으므로, 가능한 가장 간결하고 효율적으로 작성되어야 합니다.

    당신의 응답은 오직 새로운 프롬프트 텍스트만 포함해야 하며, 그 외의 내용은 없어야 합니다."""
    try:
        response = gemini_client.models.generate_content(
            model=reflector_model_name,
            contents=reflection_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=30,
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
            ),
        )
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Gemini API 오류: {str(e)}. Gemini API 키를 확인하세요.")


def select_candidate_for_mutation(candidate_pool, num_tasks):
    """파레토 기반 전략에 따라 변형할 다음 후보를 선택합니다."""
    if not candidate_pool:
        return None
    if len(candidate_pool) == 1:
        return candidate_pool[0]
    best_scores_per_task = [-1.0] * num_tasks
    for c in candidate_pool:
        for i in range(num_tasks):
            if c["scores"][i] > best_scores_per_task[i]:
                best_scores_per_task[i] = c["scores"][i]
    pareto_front_ids = {
        c["id"]
        for c in candidate_pool
        for i in range(num_tasks)
        if abs(c["scores"][i] - best_scores_per_task[i]) < 1e-6
    }
    if not pareto_front_ids:
        return max(candidate_pool, key=lambda c: c["avg_score"])
    selected_id = random.choice(list(pareto_front_ids))
    return next(c for c in candidate_pool if c["id"] == selected_id)


async def run_gepa_optimization(
    openai_key,
    gemini_key,
    model_id,
    reflector_model_name,
    seed_prompt,
    training_data,
    budget,
):
    """
    GEPA 최적화 프로세스를 총괄하는 메인 비동기 함수.
    """
    # --- 초기화 ---
    print(log_message("GEPA 최적화 프로세스를 시작합니다..."))
    # 비동기/동기 클라이언트 모두 초기화
    async_openai_client = openai.AsyncOpenAI(api_key=openai_key)
    sync_openai_client = openai.OpenAI(api_key=openai_key)
    gemini_client = genai.Client(api_key=gemini_key)

    rollout_count = 0
    candidate_pool = []
    best_candidate = {"prompt": "초기화 중...", "avg_score": -1.0}

    # --- 초기 시드 프롬프트 평가 ---
    print("\n" + "=" * 50)
    print(log_message("1단계: 초기 시드 프롬프트 병렬 평가"))
    start_time = time.time()

    scores, avg_score = await evaluate_candidate_async(
        async_openai_client, model_id, seed_prompt, training_data
    )
    rollout_count += len(training_data)

    initial_candidate = {
        "id": 0,
        "prompt": seed_prompt,
        "parentId": None,
        "scores": scores,
        "avg_score": avg_score,
    }
    candidate_pool.append(initial_candidate)
    best_candidate = initial_candidate

    end_time = time.time()
    print(
        log_message(
            f"초기 평가 완료. 소요 시간: {end_time - start_time:.2f}초", "success"
        )
    )
    print(
        log_message(
            f"시드 프롬프트 초기 점수: {initial_candidate['avg_score']:.2f}", "best"
        )
    )
    print(f"현재 최적의 프롬프트:\n---\n{best_candidate['prompt']}\n---")

    # --- 메인 최적화 루프 ---
    print("\n" + "=" * 50)
    print(log_message(f"2단계: 최적화 루프 시작 (예산: {budget} 롤아웃)"))
    while rollout_count < budget:
        print(log_message(f"--- 반복 시작 (롤아웃: {rollout_count}/{budget}) ---"))
        parent_candidate = select_candidate_for_mutation(
            candidate_pool, len(training_data)
        )

        # '반성' 단계는 동기적으로 1회만 호출
        task_index = random.randint(0, len(training_data) - 1)
        reflection_task = training_data[task_index]
        try:
            # 반성을 위한 1회 동기 호출
            rollout_output = (
                sync_openai_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": parent_candidate["prompt"]},
                        {"role": "user", "content": reflection_task["input"]},
                    ],
                    max_tokens=5,
                    temperature=0.4,
                )
                .choices[0]
                .message.content.strip()
            )
            rollout_count += 1
            eval_result = evaluation_and_feedback_function(
                rollout_output, reflection_task
            )

            # 새 프롬프트 생성
            new_prompt = reflect_and_propose_new_prompt(
                gemini_client,
                reflector_model_name,
                parent_candidate["prompt"],
                [
                    {
                        "input": reflection_task["input"],
                        "output": rollout_output,
                        "feedback": eval_result["feedback"],
                    }
                ],
            )
            print(
                log_message(
                    f"새로운 후보 프롬프트 #{len(candidate_pool)}를 생성했습니다."
                )
            )

            # 새 프롬프트 병렬 평가
            start_time = time.time()
            new_scores, new_avg_score = await evaluate_candidate_async(
                async_openai_client, model_id, new_prompt, training_data
            )
            rollout_count += len(training_data)
            end_time = time.time()
            print(
                log_message(
                    f"새 후보 평가 완료. 소요 시간: {end_time - start_time:.2f}초",
                    "success",
                )
            )

            new_candidate = {
                "id": len(candidate_pool),
                "prompt": new_prompt,
                "parentId": parent_candidate["id"],
                "scores": new_scores,
                "avg_score": new_avg_score,
            }

            if new_candidate["avg_score"] >= best_candidate["avg_score"]:
                print(
                    log_message(
                        f"새 후보 #{new_candidate['id']} 성능 향상 또는 유지! 점수: {new_candidate['avg_score']:.2f} >= {best_candidate['avg_score']:.2f}",
                        "success",
                    )
                )
                candidate_pool.append(new_candidate)
                if new_candidate["avg_score"] > best_candidate["avg_score"]:
                    best_candidate = new_candidate
                    print(log_message("새로운 최적의 프롬프트를 찾았습니다!", "best"))
                    print(
                        f"현재 최적의 프롬프트:\n---\n{best_candidate['prompt']}\n---"
                    )
            else:
                print(
                    log_message(
                        f"새 후보 #{new_candidate['id']}는 성능이 향상되지 않았습니다. 점수: {new_candidate['avg_score']:.2f}. 폐기합니다.",
                        "fail",
                    )
                )

        except Exception as e:
            print(log_message(f"최적화 반복 중 오류 발생: {str(e)}", "fail"))
            rollout_count += 1

    print("\n" + "=" * 50)
    print(log_message("최적화 예산을 모두 소진했습니다. 종료합니다.", "best"))
    print(f"최종 최적 프롬프트 (점수: {best_candidate['avg_score']:.2f}):")
    print(f"\n{best_candidate['prompt']}\n")
    print("=" * 50)
    return best_candidate


async def main():
    """메인 실행 함수"""
    # --- 설정 ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_ID = "gpt-4o-mini"
    REFLECTOR_MODEL_NAME = "gemini-2.5-pro"
    SEED_PROMPT = """당신은 뉴스 기사 분류 전문가입니다. 제시된 기사가 자동차 산업과 관련이 있는지 판단하여 숫자로만 응답하세요.

【판단 기준】
자동차 관련(1)으로 분류:
- 자동차 제조사가 주요 주체인 기사
- 자동차/전기차/수소차의 생산, 판매, 개발 내용
- 자동차 전용 부품/기술: 배터리(전기차용), 반도체(차량용), 타이어, 유리, 파워트레인, 열관리시스템, 카메라모듈(차량용), AP모듈(차량용), EVCC
- 자동차 직접 서비스: 충전소, 정비, 자동차보험, 카셰어링
- 자율주행 기술(차량 적용 명시)
- 모빌리티 서비스(UAM 포함)

자동차 무관(0)으로 분류:
- 자동차가 단순 예시로만 언급
- 범용 기술/부품(자동차 특정 언급 없음): 일반 반도체, 일반 배터리, 일반 AI/로봇
- 에너지/ESS(주택용/산업용)
- 무역정책/관세(자동차 특정 언급 없음)
- 타 산업 중심(방산, 항공, 철도 등)

【핵심 식별 패턴】
✓ 기업명+자동차 사업: 현대차, 기아, GM, 테슬라, BYD, 도요타 등
✓ 자동차 전용 키워드: 전기차, EV, 자율주행차, ADAS, SDV, 충전인프라, OEM(자동차)
✓ 부품사+자동차 공급: "차량용", "자동차용", "전기차용", "자동차 OEM 공급"

【경계 사례 판단】
- 배터리 → 전기차용 명시(1) / ESS·주택용(0)
- 반도체 → 차량용·자율주행용(1) / 일반·데이터센터용(0)
- AI/로봇 → 자율주행·차내 시스템(1) / 산업용·일반(0)
- 인프라 → 충전소·자율주행도로(1) / 일반 스마트시티(0)
- M&A/투자 → 자동차 기업·부품사(1) / 타업종(0)

【중요】
- 제목과 본문 전체를 종합 판단
- 자동차가 핵심 주제인지 확인
- 애매한 경우 본문의 주요 논점 기준
- 반드시 0 또는 1만 출력
- 설명이나 이유 없이 숫자만 응답

출력: 0 또는 1"""

    # --- CSV에서 훈련 데이터 로드 ---
    TRAINING_DATA = []
    try:
        df = pd.read_csv("./data/car_samples.csv")  # CSV 파일명
        for index, row in df.iterrows():
            input_text = f"[기사]\n\n제목: {row['title']}\n\n내용: {row['content']}"
            TRAINING_DATA.append(
                {"id": index, "input": input_text, "expected_output": str(row["label"])}
            )
        print(
            log_message(
                f"'{df.shape[0]}'개의 훈련 데이터를 CSV 파일로부터 성공적으로 로드했습니다.",
                "success",
            )
        )
    except Exception as e:
        print(log_message(f"CSV 파일 로드 실패: {e}", "fail"))
        return

    BUDGET = 250  # 예산 재설정 (초기 46 + (47 * 4 사이클) = 234)

    # --- 최적화 실행 ---
    if OPENAI_API_KEY and GEMINI_API_KEY and TRAINING_DATA:
        try:
            await run_gepa_optimization(
                openai_key=OPENAI_API_KEY,
                gemini_key=GEMINI_API_KEY,
                model_id=MODEL_ID,
                reflector_model_name=REFLECTOR_MODEL_NAME,
                seed_prompt=SEED_PROMPT,
                training_data=TRAINING_DATA,
                budget=BUDGET,
            )
        except Exception as e:
            print(f"\n실행 중 복구할 수 없는 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    asyncio.run(main())
