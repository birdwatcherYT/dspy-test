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

PROMPT = 'Given a sentence, identify and return the hobby being described.  For example, if the input is "私はギターを練習しています", the output should be "ギター".'
# 温度デフォルト： テストデータ精度: 7.14%
# 温度0.1： テストデータ精度: 10.71%
# few-shot examples追加
EXAMPLES = [
    {"sentence": "ミニ四駆を改造してレースに出ています", "hobby": "ミニ四駆"},
    {"sentence": "ボーカルトレーニングをして歌が上手くなりたいです", "hobby": "ボーカルトレーニング"},
    {"sentence": "ガーデニングをして、季節ごとの花を育てています", "hobby": "ガーデニング"},
]
# 温度デフォルト： テストデータ精度: 78.57%
# 温度0.1： テストデータ精度: 71.43%

# PROMPT = "与えられた文章を見て趣味を抽出してください。1つの単語で表現してください。"
# # 温度デフォルト： テストデータ精度: 10.71%
# # 温度0.1： テストデータ精度: 14.29%
# EXAMPLES = [
#   {"sentence": "週末はキャンプに行って焚き火を楽しんでいます", "hobby": "キャンプ"},
#   {"sentence": "バスケットボールをするのが趣味です", "hobby": "バスケットボール"},
#   {"sentence": "休日に散歩して鳥の写真を撮ります", "hobby": "バードウォッチング"}
# ]
# # few-shot examples追加
# # 温度デフォルト： テストデータ精度: 60.71%
# # 温度0.1： テストデータ精度: 60.71%

# fmt: on


def create_hobby_extraction_chain():
    """趣味抽出用のチェーンを作成"""
    llm = ChatOllama(
        model="gemma3",
        temperature=0.1,
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
        match = "✓" if prediction == label else "✗"
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
