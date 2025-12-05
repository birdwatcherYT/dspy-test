import dspy
import os
import dotenv
import json

dotenv.load_dotenv()

# LM設定
# lm = dspy.LM("gemini/gemini-2.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
lm = dspy.LM("ollama_chat/gemma3", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)


class HobbyExtraction(dspy.Signature):
    """趣味は？"""

    # ↑ここに指示を書く↑

    sentence: str = dspy.InputField(desc="入力文章")
    hobby: str = dspy.OutputField(desc="抽出された趣味")


class HobbyExtractor(dspy.Module):
    def __init__(self):
        # self.extract = dspy.ChainOfThought(HobbyExtraction)
        self.extract = dspy.Predict(HobbyExtraction)

    def forward(self, sentence):
        return self.extract(sentence=sentence)


# 使用例
extractor = HobbyExtractor()
result = extractor(sentence="休日はカフェで読書をして過ごします")
print(result.hobby)  # => "読書"
# データセット

with open("train.json", "r", encoding="utf-8") as f:
    train = json.load(f)
with open("test.json", "r", encoding="utf-8") as f:
    test = json.load(f)

# データ分割（trainを80:20でtrain/validationに分割）
all_train = [dspy.Example(**d).with_inputs("sentence") for d in train]
split_idx = int(len(all_train) * 0.8)
trainset = all_train[:split_idx]
valset = all_train[split_idx:]
testset = [dspy.Example(**d).with_inputs("sentence") for d in test]


# メトリック定義
def hobby_metric(example, pred, trace=None):
    # 完全一致または部分一致
    return example.hobby == pred.hobby
    # return example.hobby in pred.hobby or pred.hobby in example.hobby


# 最適化
# optimizer = dspy.BootstrapFewShot(
#     metric=hobby_metric, max_bootstrapped_demos=4, max_labeled_demos=2
# )
# optimized_extractor = optimizer.compile(student=HobbyExtractor(), trainset=trainset)

# MIPROv2で最適化
optimizer = dspy.MIPROv2(
    metric=hobby_metric,
    auto="light",  # light/medium/heavyから選択
    num_threads=8,
    track_stats=True,
)

optimized_extractor = optimizer.compile(
    HobbyExtractor(),
    trainset=trainset,
    valset=valset,
    max_bootstrapped_demos=3,
    max_labeled_demos=2,
)

# 最適化結果の確認
print(f"最適化スコア: {optimized_extractor.score}")
print(f"試行ログ: {optimized_extractor.trial_logs}")
# 最適化後のプログラムを保存
optimized_extractor.save("optimized_hobby_extractor.json")


# 評価
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=testset, metric=hobby_metric, display_progress=True)
score = evaluator(optimized_extractor)
print(f"テストスコア: {score}")


# 最適化されたプログラムを実行
result = optimized_extractor(sentence="合唱団に入って歌を練習しています")
print(result)
result = optimized_extractor(sentence="競馬のレースを観戦するのが趣味です")
print(result)
result = optimized_extractor(sentence="休日に散歩して鳥の写真を撮ります")
print(result)

# 最後に使用されたプロンプトを表示
dspy.inspect_history(n=3)
