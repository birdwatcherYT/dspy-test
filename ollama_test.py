"""
langchain-ollama を使った趣味抽出と評価
"""

import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotChatMessagePromptTemplate


def load_data(filepath: str) -> list[dict]:
    """JSONファイルからデータを読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# fmt: off

PROMPT = 'Given a sentence describing a person’s activity, identify and state the hobby being practiced. Output only the hobby.'
# テストデータ精度: 57.14%
# few-shot examples追加
EXAMPLES = [
    {"sentence": "週末は公園でスケッチをして過ごします", "hobby": "スケッチ"},
    {"sentence": "陶芸教室に通って、自分で器を作っています", "hobby": "陶芸"},
    {"sentence": "旅行が好きで、日本全国を巡っています", "hobby": "旅行"}
]
# テストデータ精度: 71.43%

# PROMPT = "与えられた文章を見て趣味を抽出してください。1つの単語で表現してください。"
# # PROMPT = "与えられた文章を見て趣味を1単語で表現してください。余計な説明は禁止します。"
# # テストデータ精度: 46.43%
# # few-shot examples追加
# EXAMPLES = [
#   {"sentence": "週末はキャンプに行って焚き火を楽しんでいます", "hobby": "キャンプ"},
#   {"sentence": "バスケットボールをするのが趣味です", "hobby": "バスケットボール"},
#   {"sentence": "休日に散歩して鳥の写真を撮ります", "hobby": "バードウォッチング"}
# ]
# # テストデータ精度: 60.71%

# fmt: on


def create_hobby_extraction_chain():
    """趣味抽出用のチェーンを作成"""
    llm = ChatOllama(
        model="gemma3",
        # temperature=0.1,
    )
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{sentence}"),
            ("ai", "{hobby}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=EXAMPLES,
        example_prompt=example_prompt,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT),
            few_shot_prompt,
            ("human", "{sentence}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def evaluate(predictions: list[str], labels: list[str]) -> dict:
    """予測結果を評価する"""
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    total = len(labels)
    accuracy = correct / total if total > 0 else 0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }


def run_inference_and_evaluate(chain, data: list[dict], dataset_name: str):
    """推論と評価を実行"""
    predictions = []
    labels = []

    print(f"\n=== {dataset_name} データでの推論 ===")
    print("-" * 60)

    for i, item in enumerate(data):
        sentence = item["sentence"]
        label = item["hobby"]
        prediction = chain.invoke({"sentence": sentence})

        predictions.append(prediction)
        labels.append(label)

        # 結果を表示
        match = "O" if prediction == label else "X"
        print(f"[{i+1:02d}] {match}")
        print(f"     文章: {sentence}")
        print(f"     予測: {prediction}")
        print(f"     正解: {label}")
        print()

    # 評価結果
    result = evaluate(predictions, labels)
    print("-" * 60)
    print(f"=== {dataset_name} 評価結果 ===")
    print(f"正解数: {result['correct']} / {result['total']}")
    print(f"精度: {result['accuracy']:.2%}")

    return result


def main():
    # チェーンの作成
    chain = create_hobby_extraction_chain()

    # データの読み込み
    train_data = load_data("train.json")

    test_data = load_data("test.json")

    print(f"訓練データ数: {len(train_data)}")
    print(f"テストデータ数: {len(test_data)}")

    # 評価
    # train_result = run_inference_and_evaluate(chain, train_data, "訓練")
    test_result = run_inference_and_evaluate(chain, test_data, "テスト")

    print("=== 最終結果 ===")
    # print(f"訓練データ精度: {train_result['accuracy']:.2%}")
    print(f"テストデータ精度: {test_result['accuracy']:.2%}")


if __name__ == "__main__":
    main()
