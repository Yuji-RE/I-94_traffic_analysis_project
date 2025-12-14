# %%
# ruff: noqa: E402
# %% [markdown]
"""
<a id="overview"></a>
## データの概要把握

今回使用するデータセットは、米国ミネソタ州のI-94高速道路における交通量データである。
まずは、今回使用する主要ライブラリをインポートし、データセットのCSVファイルを読み込む。

Scipyの代わりにrpy2を使う理由としては、信頼区間や自由度までデフォルトで出力してくれるため中身が確認しやすく、
自分の統計レベル（統計検定2級程度）とより合っているツールだと思ったためである。

個人的な好みではあるが、データの概要をつかむ際は、複数セルに分けて出力するよりも、
1つの出力領域にまとめた方がスクロールの手間なく全体を一目で把握できて効率的に思える。
そのため、あえて繋げて1つのセルに統合している。
"""
# %%
# 今回のプロジェクトに必要なライブラリをインポートする
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import rpy2.robjects as ro
from rpy2.robjects import FloatVector
from IPython.display import display

# この規模のプロジェクトでは必要ないが、将来的な拡張性と実務を見越して、
# プロジェクトルートを定義しておく
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "raw"
csv_path = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"

# CSVファイルのデータを読み込む
i_94 = pd.read_csv(csv_path)

# データの概要をつかむ

# 出力結果を見やすくするため、`print()`の代わりに、
# オブジェクトを綺麗に出力してくれる`display()`を採用
display(i_94.head())
display(i_94.tail())
display(i_94.describe())
i_94.info()

# %% [markdown]
"""
以上のことから、次のことが分かった。

#### 【構造】

- 行: 48204, 列: 9のデータセットである
- 2012-10-02 09:00:00 から 2018-09-30 23:00:00 までのデータが揃っている
- `traffic_volume`以外の項目は、「時間」及び「気象」関連の条件である
→ EDAは、時間的要因と気象条件と交通量の関係性を分析する方向性となる
- `holiday`→ データはほぼ欠損していて使い物にならない
- `date-time`→ データは文字列なので`datetime64型`に変換する必要がある

---

#### 【外れ値】

- `temp`→ ケルビン温度なのに最小値が0となっており、異常値だと考えられる
- `rain_1h`→ 最大値が約9800mmとなっており、外れ値だと考えられる
- `traffic_volume`→ 最小値が0となっており、深夜/早朝or交通規制orデータの欠損の可能性がある（要検証）

`temp`と`rain_1h`の外れ値について、現時点ではデータ全体の傾向を把握する段階にあるため、削除は行わない。
これらの値が後の分析結果や可視化に影響を及ぼすと判明した場合には、その時点で除外または処理方法
（スケーリング調整で外れ値による描画の歪みを回避するなど）を再検討する方針とする。

※ 外れ値の判定については、各変数の統計量（特に四分位数や平均値など）を踏まえつつ、常識的に考えても明らかに極端な値を
ここでは「外れ値」と推測している。

---

##### 【分布】

`traffic_volume`
1. 平均値(3259)と中央値(3380)でズレが小さいので、歪みは小さいと考えられる
2. 標準偏差(1986)が平均値の約60%を占めており、変動がかなり大きいと考えられる

※ `traffic_volume`を目的変数とし、他の「時間」や「気象条件」を示すデータを説明変数として分析する。<br>
よって、分布に着目すべきデータは`traffic_volume`カラムのデータであることがわかる。
"""
# %% [markdown]
"""
<a id="processing"></a>
## データの前処理

まずは、日付データが文字列になっているので、後で混乱しないように早めに `datetime64型`に変換する。
この際、日付データは既に現地時間で集計されているため、UTCによる標準化→現地時間変換の作業は必要ない。

また、本分析の主体となる`traffic_volume`について、分布を視覚的に捉えるために、ヒストグラムを作成する。
"""
# %%
# 日付データを分析（抽出やグループ化など）可能な形に変換する
i_94["date_time"] = pd.to_datetime(i_94["date_time"])

# 今回使用する時系列データを要素ごとに分解して後で分解しやすくする
i_94["date_only"] = i_94["date_time"].dt.date
i_94["hour"] = i_94["date_time"].dt.hour
i_94["day"] = i_94["date_time"].dt.day
i_94["month"] = i_94["date_time"].dt.month
i_94["year"] = i_94["date_time"].dt.year

# %%
# 交通量の度数分布をヒストグラムで可視化する

# デフォルトだと階級幅とbin境界がズレて見にくいので、調整する
bin_width = 1000

xmin = 0
x_max = i_94["traffic_volume"].max()
bins = np.arange(xmin, x_max + bin_width, bin_width)

ax = i_94["traffic_volume"].plot.hist(bins=bins, color="skyblue", edgecolor="black")

ax.set_xticks(bins)

plt.grid(axis="y", linestyle="--", color="gray", alpha=0.5)
plt.title("I-94 Traffic Volume")
plt.xlabel("Traffic Volume  (Average Vehicles per Hour)")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
"""
`traffic_volume`のヒストグラムの結果から、二峰性が伺える。

0～1000台と4000～5000台の階級で度数が特に増えていることから、
「交通量が増えるラッシュアワー」と「交通量が減る閑静期」の2つの分布が混ざっている可能性を疑う。

`traffic_volume`と他の説明変数との相関係数を確認して、まずはどの要因が交通量に影響を与えているのかを調べていく。
"""
# %%
i_94.corr(numeric_only=True)["traffic_volume"].sort_values(ascending=False)

# %% [markdown]
"""
データセットのうち、量的なデータを扱うコラムの中では`traffic_volume`と相関がありそうなのは`hour`だということがわかった。

次に、一番交通量と相関が強そうな日付データの`hour`項目について分析していく。
最初のステップとして「時間帯によって交通量が変動する」のであれば、「日中」と「夜間」での交通量の分布の違いを確認する。
"""
# %% [markdown]
"""
#### 日中と夜間の交通量の分布比較

両者の分布を知るにあたって、前処理として`i_94`データセットを2つの時間帯「日中」と「夜間」のデータセットに分割していく。
そのあと、作成した2つのヒストグラムを比較対照する。
"""
# %%
# 日中データと夜間データに分割
# 元データ`i_94`に不本意な影響を及ぼさないよう、`copy()`をつかう
day_time = i_94[(i_94["hour"] >= 6) & (i_94["hour"] <= 18)].copy()
night_time = i_94[(i_94["hour"] > 18) | (i_94["hour"] < 6)].copy()

# 正しく元データを二つに分割できてるか確認（確認作業1）
print(day_time.shape)
print(night_time.shape)

# %%
# 2つのデータの行和が元データの行総数に照合するかを確認
# → `True`と出力されればOK（確認作業2）
day_time.shape[0] + night_time.shape[0] == len(i_94)

# %% [markdown]
"""
二つの確認作業によって、データが正しく分割されていることが確認できたので、
早速両者の分布をヒストグラムで可視化して比較対照していく。
"""
# %%
# 新たに作成した「日中データ」と「夜間データ」の度数分布をヒストグラムで可視化

# 全体のキャンバスを作成
plt.figure(figsize=(11, 4.5))

# 日中データについての描画
plt.subplot(1, 2, 1)
plt.hist(day_time["traffic_volume"], color="orange", edgecolor="black")
plt.grid(axis="y", linestyle="--", color="gray", alpha=0.5)
plt.xlim(0, 8000)
plt.ylim(0, 8500)
plt.title("I-94 Traffic Volume (Daytime)")
plt.xlabel("Traffic Volume  (Average Vehicles per Hour)")
plt.ylabel("Frequency")

# 夜間データについての描画
plt.subplot(1, 2, 2)
plt.hist(night_time["traffic_volume"], color="blue", edgecolor="black")
plt.grid(axis="y", linestyle="--", color="gray", alpha=0.5)
plt.xlim(0, 8000)
plt.ylim(0, 8500)
plt.title("I-94 Traffic Volume (Nighttime)")
plt.xlabel("Traffic Volume  (Average Vehicles per Hour)")
plt.ylabel("Frequency")

# %% [markdown]
"""
- 日中の交通量の分布は左裾が伸びている
- 夜間の交通量の分布は右裾が伸びている

→ 両者の分布形状から、同一性は否定される。したがって、「日中」と「夜間」では交通量の分布に有意な差があると考えられる。
ただし、ヒストグラムで把握できるのは分布の大まかな傾向にとどまるため、より精緻に確認するために統計量も併せて検討する。
"""
# %%
# 統計量の対称比較がしやすいように、横並びレイアウトとなるような要約表を
# 'pd.DataFrame()'で作成
summary = pd.DataFrame(
    {
        "Daytime": day_time["traffic_volume"].describe(),
        "Nighttime": night_time["traffic_volume"].describe(),
    }
)

print("Traffic Volume")
display(summary)

# %% [markdown]
"""
日中の方が交通量が多いことが明示された。<br>
今回の分析目的は「交通量の多い時間帯を見つける事」なので、<br>
これからは日中の交通量データに絞って、更なる要因の発見に努めていく。
"""
# %% [markdown]
"""
## Time Indicators

これより、交通量との相関が示唆された「時間的要因」についてさらなる深堀を行っていく。
"""
# %% [markdown]
"""
### Relationship between Month and Traffic Volume

まずは、季節変動を掴むために、月ごとの交通量の平均を求めて折れ線グラフで表示する。
"""
# %%

# 日中データに限って、全年度を通した月ごとの平均交通量を計算
by_month = day_time.groupby("month")["traffic_volume"].mean().sort_index()
# グラフを描画
by_month.plot(marker="o", color="green", linewidth=2)
plt.title("Daytime Average Traffic Volume by Month", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Average Traffic Volume  (vehicles per hour)")
plt.grid(True, linestyle="--", alpha=0.6)

# %% [markdown]
"""
春から秋にかけて交通量がピークを迎え、冬の間交通量が急降下してボトムとなる周期性を捉えることができた。
今回の分析目的は「交通量の多いゴールデンタイム」を特定することなので、特に交通量の多い春～秋の時期に今後はフォーカスしていく。

しかし、その前に７月頃に発生した不自然な急降下についても調べたい。

この原因を知ることによって、交通量の多い時期の中でも地雷となる期間を避けてキャンペーンを施行できるようになるかもしれないし、
最初に明らかになった謎の「交通量最小値: 0」の真相を掴めるかもしれないからだ。
"""
# %% [markdown]
"""
#### 不振な推移（7月の交通量の急降下）の原因究明

これから、「暖かい時期に交通量が増えるという周期性がある」という仮説に矛盾する、7月における交通量の不自然な減少について
調べていく。まずは、異常値の影響の有無を確認するために、IQRを用いて外れ値を除外した後の月ごとの平均交通量を計算し、
元の月ごとの平均交通量と比較する。
"""
# %%

# 外れ値判定の幅を決定する定数kを設定
k = 1.5

# 欠損値を除外した日中データを作成
day_time_clean = day_time.dropna(subset=["month", "traffic_volume"]).copy()

# IQRを用いて外れ値を除外
g = day_time_clean.groupby("month")["traffic_volume"]
q1 = g.transform(lambda s: s.quantile(0.25))
q3 = g.transform(lambda s: s.quantile(0.75))
iqr = q3 - q1

lower = q1 - k * iqr
upper = q3 + k * iqr

day_time_no_out = day_time_clean[
    (day_time_clean["traffic_volume"] >= lower)
    & (day_time_clean["traffic_volume"] <= upper)
]

# 月ごとの平均交通量を計算（外れ値除外後と元データの両方）
by_month_iqr = day_time_no_out.groupby("month")["traffic_volume"].mean().sort_index()
by_month = day_time.groupby("month")["traffic_volume"].mean().sort_index()

# グラフを描画して比較
ax = by_month.plot(marker="o", linewidth=2, label="Mean (original)")
by_month_iqr.plot(ax=ax, marker="o", linewidth=2, label="Mean (IQR outliers removed)")

plt.title("Average Traffic Volume by Month", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Average Traffic Volume (vehicles per hour)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# %% [markdown]
"""
外れ値を除外した月ごとの平均交通量は、オリジナルに比べて全体的に上昇しているが、大まかなパターンは変わっていない。
外れ値の影響を受けやすいオリジナルのデータの方が全体的に低い値をとっていることから、外れ値は「データの欠損系」である
可能性が考えられる。しかし、どちらの場合であっても、依然として7月に不自然な急降下が発生していることに変わりはない。

次に、
各年の七月についての交通量をまとめたグラフを作成し、より具体的に「いつ頃にこの不自然な急降下が発生したのか」を調べていく。
"""
# %%
# 7月の交通量を年ごとに平均して可視化する（日中データ）
by_year_july = (
    day_time.query("month == 7")  # 日中データにおいて、7月だけ抽出
    .groupby("year")["traffic_volume"]  # 年ごとに交通量をグループ化
    .mean()  # 各年の平均交通量の平均を計算（平均の平均）
    .sort_index()
)

# グラフ描画
by_year_july.plot(marker="o", color="green", linewidth=2)
plt.title("Average July Traffic Volume by Year", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Average Traffic Volume")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
どうやら、2016年の7月におけるなんらかの原因で、7月全体の交通量が下がっていたという可能性が見えてきた。

そのため、これから2016年の7月の何日にそれが起こったのかを調べてみる。
"""
# %%

# 2016年7月に限定した日ごとの平均交通量を計算
by_day_2016_july = (
    day_time.query("year == 2016 and month == 7")  # 2016年において7月だけ抽出
    .groupby("day")["traffic_volume"]  # 日単位でグループ化
    .mean()  # 各日の平均を計算
)

# グラフを描画
plt.figure(figsize=(12, 5))
by_day_2016_july.plot(marker="o", color="green", linewidth=2)
plt.title("Average Traffic Volume in July 2016 (by Day)", fontsize=14)
plt.xlabel("Day of Month")
plt.ylabel("Average Traffic Volume")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(range(1, 32))  # X軸の目盛りを1日ごとに設定
plt.tight_layout()
plt.show()

# %% [markdown]
"""
この結果から、2016年における7月の不自然な交通量の急行落は、23日から24日にかけてのデータに強い影響を受けていることがわかる。

また、ウェブリサーチの結果、以下の2つのことが分かった。
- 4日→独立記念日（祝日）による交通量減少の可能性<sup><a href="#fn1" id="ref1" style="color: red;">1</a></sup>
- 9,10日→大規模な抗議活動による閉鎖発生による影響の可能性<sup><a href="#fn2" id="ref2" style="color: red;">2</a></sup>
<br>

このようなイベント発生日よりも交通量が大幅に下がっていることを考慮すると、俄然23日~24日の交通量は不自然であるように思える。
この時点で、外れ値であることは自明である。
<br><br>
<hr>

<p id="fn1">
  <a href="#ref1" style="color: red;">1.</a>
  <a href="https://content.govdelivery.com/accounts/USDHSCBP/bulletins/12baf00#:~:text=05%20PM%20EST-,Air%20Manifest,Monday%2C%20December%2026%20Christmas%20Day">2016年の連邦祝日</a>のウェブページを参照(最終アクセス日: 2025年12月11日)
</p>

<p id="fn2">
  <a href="#ref2" style="color: red;">2.</a>
  <a href="https://www.twincities.com/2016/07/15/how-the-i-94-takeover-became-a-full-scale-riot/">
PIONEER PRESSのニュース記事</a>を参照(最終アクセス日: 2025年12月11日)
</p>
"""
# %% [markdown]
"""
そこで今度は、より詳細なグラスプを得るために、当年の7月22~27日のデータについて比較対照を行っていく。
日数の選定基準としては、目算で正常なデータと異常なデータを視覚的に捉えられる範囲を適当に決定した。

今回の可視化では、グラフの不振な動きを見逃さずに全体的な推移を捉えられるよう、
`day_time`（日中データ）ではなく`i_94`（全日データ）で比較する。
"""
# %%
# 7月22〜25日のデータを抽出して、時間ごとの平均を算出

by_hour_july_22_25 = (
    i_94.query("year == 2016 and month == 7 and day in [22,23,24,25,26,27]")
    .groupby(["day", "hour"])["traffic_volume"]
    .mean()
    .unstack("day")
)

# グラフ描画
plt.figure(figsize=(12, 5))
by_hour_july_22_25.plot(marker="o", linewidth=2)
plt.title("Average Traffic Volume (July 22–25)", fontsize=14)
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Average Traffic Volume", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Day")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
23日と24日は他の日と比べて不自然な推移をしているため、臨時的な交通止めもしくはデータの欠損の可能性が考えられる。
とくに、23日の外れ値に関しては、22日の23:00頃から24日の4:00まで続き、その余韻ともとれる不審な変動が同日の16:00まで続いている。

どちらにせよ、この急降下は偶発的な要因によるものと予想され、現時点では特定の時間帯との因果は見られない。
もしかすると気象条件と交通量の相関調査で、この不自然な推移について新たに明らかになることがあるかもしれないため、
この外れ値はまだデータセットから削除せず、そのまま分析を続行していく（頭の片隅にこの外れ値のことは常においとく）。

それでは、7月の不審な推移の調査に一段落が付いたところで、本プロジェクトの主目的「いつ交通量が最も多くなるのか」に舵を戻す。
"""
# %% [markdown]
"""
### Relationship between Day of the Week and Traffic Volume

現時点で、「月（季節）と交通量の関係」がわかっているので、次のステップとして、「曜日と交通量の関係性」について調べていく。

曜日は7項目（月～日）あるので、同一のキャンバスに描くとぐちゃぐちゃして見にくい。かといってすべて異なるキャンバスに描いても
比較対照しずらい。
解決策としては、曜日の中でも「平日」「週末」に分割して、それぞれの平均を算出して比較することだが、個人的には最初は全体像を
つかみたいので、結局一つのキャンバスに全ての曜日を表示させる方針でいく。

しかし、すこしでも比較しやすくするために、各曜日の平均交通量は移動平均で平滑化し、更に凡例を平均交通量が多い順で
表示することで、全体像をつかみながらも、各曜日の交通量との関係も捉えられるようにする。
"""
# %%

# 月曜日を始点として、各週（月～金）ごとにグループ化
# 各データの曜日（0=Mon,...,6=Sun）を取得
# '各週 × 曜日ごと'に平均交通量の平均を算出
# '曜日'を行からindexに展開
# 曜日を数字から英語表記に変換
# 正しい時間軸に並べ替え
by_week_weekday = (
    day_time.groupby(
        [
            pd.Grouper(key="date_time", freq="W-MON"),
            day_time["date_time"].dt.dayofweek,
        ]
    )["traffic_volume"]
    .mean()
    .unstack()
    .rename(
        columns={0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )
    .sort_index()
)

# 複数の時系列データを比較対照したいため、見やすいように平滑化する（4週移動平均で算出）
by_week_weekday_MA = by_week_weekday.rolling(window=4, min_periods=1).mean()


# 各曜日の平均交通量を算出して、降順に並べる
weekday_order = by_week_weekday_MA.mean().sort_values(ascending=False).index.tolist()
# グラフを描画
ax = by_week_weekday_MA.plot(figsize=(12, 6), linewidth=2)

# 凡例を、降順の曜日順で表示
ax.legend(title="date (sorted)", labels=weekday_order)

ax.set_title(
    "Daytime Average Traffic Volume by Weekday (Weekly, 4-week MA)", fontsize=14
)
ax.set_xlabel("Week")
ax.set_ylabel("Average Traffic Volume (Vehicles per hour)")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
この結果から、交通量に関して、平日＞土日の傾向があること、木曜・水曜・金曜の順番で交通量が多いことを確認できた。
しかし、懸念した通りグラフが直感的に捉えずらく、特に交通量の分布が拮抗している平日の中で「どの曜日がどれくらいの差で交通量が
多いのか」が依然としてわからない。よって、このグラフを簡易化し、曜日ごとに平均を算出したものプロットする必要がありそうだ。

ただ、全体像を表示したことによって、またもやデータの不審な動きをとらえることができた。2014年の夏ごろから2015年の夏ごろに
かけて、データが全く変動していない。

原因としてはこの間のデータの欠損が予想されるが、念のために、まずはこの現象の原因究明を試みる。
"""
# %% [markdown]
"""
#### 変動しないデータの正体

既知の情報として、2014年と2015年に変動しないデータが存在していることがわかっている。<br>
未知の情報としては月と日なので、それを特定できるように、「○ 月の平均交通量の推移」がわかるコード<br>

```python
by_day_「年を代入」_「月を代入」 = (
    day_time.loc[
        (day_time['year'] == 「年を代入」) & (day_time['month'] == 「月を代入」 )
    ]
    .groupby('day')['traffic_volume']
    .mean()
)
```
<br>

を2つ作り、`「月を代入」`のところに順次月を代入していき、グラフの出力結果を基に異常が発生した期間を
特定する原始的な方法をとる。
"""
# %%
# 2016年8月に限定した日ごとの平均交通量を計算
by_day_2014_august = (
    day_time.loc[(day_time["year"] == 2014) & (day_time["month"] == 8)]
    .groupby("day")["traffic_volume"]
    .mean()
)


# 2015年6に限定した日ごとの平均交通量を計算
by_day_2015_july = (
    day_time.loc[(day_time["year"] == 2015) & (day_time["month"] == 6)]
    .groupby("day")["traffic_volume"]
    .mean()
)

# グラフを描画
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
by_day_2014_august.plot(marker="o", color="darkorange", linewidth=2)
plt.title("Average Traffic Volume in Augost 2014 (by Day)", fontsize=14)
plt.xlabel("Day of Month")
plt.ylabel("Average Traffic Volume")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(range(1, 32))  # X軸の目盛りを1日ごとに設定
plt.tight_layout()

plt.subplot(1, 2, 2)
by_day_2015_july.plot(marker="o", color="darkblue", linewidth=2)
plt.title("Average Traffic Volume in Jun 2015 (by Day)", fontsize=14)
plt.xlabel("Day of Month")
plt.ylabel("Average Traffic Volume")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(range(1, 32))
plt.tight_layout()

# %% [markdown]
"""
分析の結果、2014年8月7日〜2015年6月12日の期間にかけて継続的な欠損があることが分かった。
よって、セル［15］の移動平均グラフで見られた「一定値が続くフラットな区間」は、この欠損によるものである可能性が高い。

本プロジェクトでは、この欠損値に対して積極的な補完や削除は行わないこととした。
理由は以下の通りである。

* 本プロジェクトで扱う日付データのサンプルサイズは十分に大きい
* 欠損が発生している期間は、特定の月や曜日に偏っておらず、交通量の季節性・曜日性の傾向を大きく歪める可能性は低いと判断される

以上から、現時点では分析結果への影響は限定的とみなし、EDAフェーズにおいては欠損値をそのまま許容する。

なお、より厳密なモデル構築や予測を目的とした分析であれば、
このような欠損期間については、適切な補完や除外ルールを設けて処理する必要がある。
"""
# %% [markdown]
"""
### Relationship between Time of the Day and Traffic Volume

気を取り直して、曜日ごとの平均交通量をプロットしたグラフを作成し、「どの曜日に一番交通量が多くなるのか」
という問いに答えていく。

セル［15］のグラフ結果は平滑化により変動が抑えられており、曜日ごとの平均交通量の差が視覚的に捉えにくい。
そこでこれから、平均交通量が最大値を記録する地点を赤色のマーカーと罫線で強調表示し、「最多平均交通量をもつ曜日」が
他の曜日と比べてどれくらい突出しているのかを直感的に把握できるようにする。
"""
# %%
# 曜日ごとの平均交通量を算出して、最も交通量が多い曜日を強調表示するグラフを描く

# 曜日ごとの比較を可能にするため、日付データから'曜日情報（0=Mon,...,6=Sun）'を取得
day_time["dayofweek"] = day_time["date_time"].dt.dayofweek

# `traffic_volume`以外の文字列データを含むカラムもグループ化してしまっているので、
# 平均を算出する際に`numeric_only=True`とする
by_dayofweek = day_time.groupby("dayofweek").mean(numeric_only=True)

# ↑で取得した'曜日情報'を、数字表記（0~6）から英語表記（Mon~Sun）に変換
day_mapping = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
by_dayofweek.rename(index=day_mapping, inplace=True)


max_volume = by_dayofweek["traffic_volume"].max()  # 曜日単位おける交通量の最大値を取得
max_day = by_dayofweek["traffic_volume"].idxmax()  # ↑の最大値を持つ曜日名を取得

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(
    by_dayofweek.loc["Mon":"Sat", "traffic_volume"],
    marker="o",
    color="orange",
    label="Weekday",
)
plt.plot(
    by_dayofweek.loc["Sat":"Sun", "traffic_volume"],
    marker="o",
    color="blue",
    label="Weekend",
)

plt.scatter(
    x=max_day,
    y=max_volume,
    color="red",
    s=60,
    zorder=3,  # 最大値を明示しつつ赤点マーカーで表示
    label=f"Peak: {max_day} ({max_volume:.0f})",
)
plt.axvline(
    x=max_day, color="red", linestyle="--", linewidth=1, alpha=0.7
)  # 赤点で交わる罫線を追加し見やすくする
plt.axhline(y=max_volume, color="red", linestyle="--", linewidth=1, alpha=0.7)

plt.title("Average Daytime Traffic Volume by Day of the Week", fontsize=16)
plt.xlabel("Day of the Week", fontsize=12)
plt.ylabel("Average Traffic Volume", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)

plt.legend(loc="upper right", bbox_to_anchor=(1, 0.8))

plt.show()

# %% [markdown]
"""
このグラフから、木曜日が最も交通量が多い曜日であり、水曜日と金曜日がそれに僅差で続いていることがわかった。
また、平日の中でも、月曜日がもっとも交通量が少ない曜日であることもわかった。

この結果から、ビルボード広告のキャンペーンを実施するならば、平日の中でも、月曜日の実施に割くコストを減らし、
木曜日・水曜日・金曜日の順番で優先的に予算を割くことが効果的であると考えられる。
"""
# %% [markdown]
"""
### Relationship between Hour of the Day and Traffic Volume

次は、一日の中でも特に交通量が多い時間帯を特定するために、時間帯ごとの平均交通量を比較していく。
これまでの分析で平日の方が交通量が多いことが分かっているため、本来は平日のグラフのみをプロットすればよいが、
より包括的な理解を得るために、週末のグラフも併せて描画して比較対照していく。
なお、このグラフでも同様に、ピーク時を赤点と罫線で強調表示し、直感的に把握しやすいようにする。
"""
# %%
# 平日・週末それぞれの時間帯ごとの平均交通量を比較し、ピークを可視化

# 平日・週末でデータを分割
business_days = day_time[day_time["dayofweek"] <= 4].copy()
weekend = day_time[day_time["dayofweek"] >= 5].copy()

# 平日の平均交通量を算出
by_hour_business = business_days.groupby("hour").mean(numeric_only=True)
by_hour_weekend = weekend.groupby("hour").mean(numeric_only=True)

# 各グループのピーク時（最大値）を取得
peak_hour_business = by_hour_business["traffic_volume"].idxmax()
peak_value_business = by_hour_business["traffic_volume"].max()


# グラフを描画
plt.figure(figsize=(12, 6))

# 平日・週末の折れ線グラフ
plt.plot(
    by_hour_business.index,
    by_hour_business["traffic_volume"],
    label=f"Weekdays (Peak={peak_value_business:.0f} at {peak_hour_business}:00)",
    color="orange",
    linewidth=2,
)
plt.plot(
    by_hour_weekend.index,
    by_hour_weekend["traffic_volume"],
    color="steelblue",
    linewidth=2,
)

# 平日のピークを赤点と破線で強調
plt.scatter(peak_hour_business, peak_value_business, color="red", s=60, zorder=3)
plt.axvline(x=peak_hour_business, color="red", linestyle="--", linewidth=1, alpha=0.7)
plt.axhline(y=peak_value_business, color="red", linestyle="--", linewidth=1, alpha=0.7)


# 軸・タイトル
plt.title("Average Traffic Volume by Hour (Daytime)", fontsize=16)
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Average Traffic Volume", fontsize=12)

plt.xlim(6, 17)
plt.xticks(ticks=range(6, 19))
plt.legend(loc="upper left", bbox_to_anchor=(0.73, 1.08))
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
このグラフから、平日の交通量は朝の6時から7時、また夕方の3時から5時にかけてラッシュアワーを迎え、
午後4時ごろにピークを迎えることがわかった。

一方で、先ほどのグラフでも示唆された通り、週末の交通量は平日に比べて全体的に低く、やはり広告効果を最大化するためには
平日にキャンペーンを実施することが望ましいと考えられる。
"""
# %% [markdown]
"""
### Quick Summary: Temporal Indicators and Traffic Volume

これまでの時間的要因について分析した結果をまとめると、次のようなことが分かった。

- 寒い季節に比べ、暖かい季節の方が交通量が多い傾向にある
- 週末に比べ、平日の方が交通量が多い傾向にある
- 平日については朝の6時から7時、また、夕方の3時から5時がラッシュアワー（午後4時ごろが最大）であり、
僅差で木・水・金の順番で交通量が多い（木曜日が最大、月曜日が最小）。
"""
# %% [markdown]
"""
## Weather Conditions

これまでは、`date_time`に関して、交通量がピークを迎える傾向にある条件を特定した。<br>
これからは、これからは気象条件を軸に交通量との関係性を分析していく。
"""
# %% [markdown]
"""
### Relationship between Temperature and Traffic Volume

まずは、量的データを扱うカラムである`temp`（気温）と交通量の関係性を調べていく。
セル［5］で出力したデータセットの量的データにおける交通量との相関係数のうち、`temp`は約0.1と相関が弱いが、
量的データにおける気象カテゴリの中ではトップの項目なので、念のために可視化してみる。

応答変数の`traffic_volume`をy軸に、説明変数の`temp`をx軸において、散布図を作成する。
"""
# %%

# 交通量と気温の散布図を作成
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(10, 6))

ax = sns.scatterplot(data=day_time, x="temp", y="traffic_volume")

plt.title("Traffic Volume vs Temperature by Weekday")
plt.xlabel("Temperature")
plt.ylabel("Traffic Volume")
# 気温データにはケルビン0℃（絶対零度）の外れ値が含まれているので、調整した範囲を設定
plt.xlim(230, 320)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
やはり交通量と気温（ケルビン温度）の間に相関は見られない。

よって、これにて量的データをあつかうカラムに関する分析は終了し、質的データを扱うカラムについての分析に移行する。
"""
# %% [markdown]
"""
### Relationship between Weather and Traffic Volume
気象条件に関して、質的データを扱うカラムは二つある。

- `wheather_main`: 天気の種類（大まかな分類）
- `weather_description`: 天気の詳細説明（細かな分類）

よって、まずは天気の種類と交通量の関係性をグラフで視覚化していく。

今回は天気の種類（カテゴリカルデータ）の度数を把握したいので、ヒストグラムではなく棒グラフを用いる。
それをなるべく見やすいように度数の多い順に並べ替え、かつ最大値が一目でわかるように対応するバーの色を変え、
具体的な値も表示する。
"""
# %%

# 天気の種類（例：Clear、Rain、Snowなど）ごとに平均交通量を比較し、
# 最も交通量が多い天気を強調表示する棒グラフを作成


# `weather_main`列における平均交通量を計算し、値をソートして取得
by_weather_main = (
    day_time.groupby("weather_main")["traffic_volume"].mean().sort_values()
)

# ↑ で取得したもののうち、最大のものを取得し、その曜日情報を取得する
max_volume_main = by_weather_main.max()
max_weather_main = by_weather_main.idxmax()


# グラフを描画
plt.figure(figsize=(10, 7))

# 【見やすい設定】
# 横棒に関して、ピーク値のものを青、それ以外を緑色で表示
colors = [
    "blue" if weather == max_weather_main else "green"
    for weather in by_weather_main.index
]
by_weather_main.plot.barh(color=colors)

# ピーク値に対応するタテの罫線を追加し見やすくする
plt.axvline(
    x=max_volume_main,
    color="red",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label=f"Peak: {max_volume_main:.0f}",
)

# 天気カテゴリの中で、最大値に対応する横棒のヨコに数字を強調して描画
y_labels = by_weather_main.index.to_list()
max_weather_position = y_labels.index(max_weather_main)
plt.text(
    max_volume_main + 50,
    max_weather_position,
    f"{max_volume_main:.0f}",
    va="center",
    color="red",
    fontweight="bold",
)


plt.title("Average Traffic Volume by Main Weather Category", fontsize=16)
plt.xlabel("Average Traffic Volume", fontsize=12)
plt.ylabel("Main Weather Category", fontsize=12)
plt.grid(True, axis="x", linestyle="--", alpha=0.6)
plt.legend(loc="upper left", bbox_to_anchor=(1, 0.90))
plt.tight_layout()
plt.show()

# %% [markdown]
"""
この結果からは、天気の種類と交通量との間に特別な関係性は示唆されない。

しかし、例えばヒストグラムでbin設定（階級の幅）が大きすぎると細かい分布の凹凸が均され、
変動が過小評価されてデータの特徴を見逃してしまうのと同じような現象が起こっている可能性もあるので、
ここで天気と交通量の間に相関や因果関係が成り立たないと結論づけるのは早計である。

なので、次に、今回分析した`wheather_main`よりも粒度の高い`wheather_description`についても、
同じ要領で棒グラフを作成し度数を再比較してみる。
"""
# %%

# 天気の詳細説明ごとに平均交通量を比較し、最も交通量が多い気象条件を強調するグラフを作成


# `weather_description`列における平均交通量を計算し、値の小さい順に取得
by_weather_description = (
    day_time.groupby("weather_description")["traffic_volume"].mean().sort_values()
)

# ↑で取得したもののうち、最大のものを取得し、その曜日情報を取得する
max_volume = by_weather_description.max()
max_weather = by_weather_description.idxmax()


# グラフの描画
plt.figure(figsize=(15, 10))

colors = [
    "blue" if weather == max_weather else "green"
    for weather in by_weather_description.index
]
by_weather_description.plot.barh(color=colors)

plt.axvline(
    x=max_volume,
    color="red",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label=f"Peak: {max_volume:.0f}",
)

y_labels = by_weather_description.index.to_list()
max_weather_position = y_labels.index(max_weather)
plt.text(
    max_volume + 50,
    max_weather_position,
    f"{max_volume:.0f}",
    va="center",
    color="red",
    fontweight="bold",
)

plt.title("Average Traffic Volume by Weather Description (Daytime)", fontsize=16)
plt.xlabel("Average Traffic Volume", fontsize=12)
plt.ylabel("Weather Description", fontsize=12)

plt.grid(True, axis="x", linestyle="--", alpha=0.6)
plt.legend(loc="upper left", bbox_to_anchor=(1, 0.9))
plt.tight_layout()
plt.show()

# %% [markdown]
"""
より細かい気象条件で見ると、天気によって交通量に差がある様子がうかがえる。
一見すると、晴れの日よりも雨・雪・曇りの日の方が交通量が多いように見える点は興味深い。
ただし、`Sky is Clear`（6行目）と`sky is clear`（16行目）が実質同じ「快晴」であるにもかかわらず別々にカウントされているため、
快晴の影響はやや過小評価されている可能性がある。
それを踏まえても、依然として「天気が良い日＝交通量が増える」とは断言できそうにない結果だ。

雨や雪の日には、ふだん徒歩や自転車で移動している人が車に切り替える可能性があり、
それに起因する渋滞が結果的に交通量を押し上げることも考えられる。
一方で、セル［10］で平均交通量の季節変動が示唆されたように、冬場は全体として交通量が少ない傾向があるにもかかわらず、
寒い時期に発生する `shower snow` や `light rain and snow` が交通量上位に入っているのは不思議な点である。

ミネソタ州周辺がもつ厳しい大陸性気候<sup><a href="#fn1" id="ref1" style="color: red;">1</a></sup>
を考慮すると、それらの気象条件が発生する時期にはそもそも交通量が少ないはずなので、
この点についてはまだ深堀り必要になってきそうだ。

また、ここで注目すべきことは「雷雨のときとスコールのときに交通量が下がるかもしれない」という懸念点である。
なぜこれが懸念点かというと、雷雨およびスコールが主に夏場に発生しやすい現象であるからだ。これまでの分析の結果、
「暖かい季節に交通量が増えやすい」という洞察を得たわけだが、その期間であっても雷雨とスコールが発生しやすい時期が
「地雷期間」の可能性があるため、それらの時期を避けてビルボードを展示するのが最も保守的な戦略となるだろう。

ということで、次は、「地雷期間＝暖かい季節でも避けたほうが良い時期」を特定していく。


ちなみに、`thunderstorm with drizzle`の時に交通量が最も少ないことと、セル［14］にて検知された23日の交通量観測の
異常の間に何か繋がりがあるのではないかと思い個人的にウェブリサーチして見た結果、その当日雷雨であったことが可能性
<sup><a href="#fn1" id="ref1" style="color: red;">2</a></sup>がでてきた。もしかすると落雷などによる観測器の異常かもしれない。

これから行っていく分析で、`thunderstorm with drizzle`が発生した日付が2016/7/23であればビンゴとなるので、
このことも念頭に置いておく。

<hr>

<p id="fn1">
    <a href="#ref1" style="color: red;">1.</a>
<a href="https://ja.climate-data.org/%E5%8C%97%E3%82%A2%E3%83%A1%E3%83%AA%E3%82%AB/%E3%82%A2%E3%83%A1%E3%83%AA%E3%82%AB%E5%90%88%E8%A1%86%E5%9B%BD/%E3%83%9F%E3%83%8D%E3%82%BD%E3%82%BF%E5%B7%9E-937/">Climate-Data.orgのウェブページ</a>よりミネソタ州の気候区分を参照
(最終アクセス日: 2025/12/11)
</p>

<p id="fn2">
    <a href="#ref2" style="color: red;">2.</a>
    <a href="https://www.weather.gov/arx/jul2316">NATIONAL WEATHER SERVICEのウェブページ</a>より2017/07/23の
ミネソタ州の天気を参照(最終アクセス日: 2025/12/11)
</p>
"""
# %% [markdown]
"""
それでは、暖かい時期（4月から10月）の中でも、避けた方がいい雷雨やスコールが発生しやすい時期を特定していく。
まずは対象の気象条件が発生した日付一覧を表示する表を作成するが、ついでにさきほど不自然な結果となった
`shower snow`と`sky is clear`についても表示できるようにする。

これにより、本プロジェクトの目的から逸れずに同時に不自然な
数字の動きにも目を配ることができる。
"""
# %%
# 天候の対象条件をリスト化
target_weathers = [
    "squalls",
    "thunderstorm with drizzle",
    "shower snow",
    "sky is clear",
]

# 表記統一と集計
day_time["weather_description"] = (
    day_time["weather_description"].str.strip().str.lower()
)

# 「暖かい時期」という条件作成
month_condition = day_time["date_time"].dt.month.between(4, 10)

# 天候の条件と月の条件を両方満たすデータを抽出
filtered = day_time[
    day_time["weather_description"].isin(target_weathers) & month_condition
]

# グループ化してリストに
grouped = (
    filtered.groupby("weather_description")["date_time"]
    .apply(list)
    .reindex(target_weathers)
    .apply(lambda x: x if isinstance(x, list) else [])
)

# 表示用のデータフレーム作成
df_display = pd.DataFrame(
    {weather: pd.Series(timestamps) for weather, timestamps in grouped.items()}
)


pd.reset_option("display.max_rows")

# デフォルトの行数で表示
display(df_display)

# %% [markdown]
"""
ここで、`sky is clear`カラム以外のカラムがほとんど欠損値で占められていて使い物にならない可能性が浮上した。
この発見が氷山の一角かもしれないので、念のために、一年間の日中データにおける各気象条件の度数を棒グラフで確認してみる。

<hr>
余談だが、上の表の1行目で`thunderstorm with drizzle`の発生日が2016年7月23日と合致したので、やはりセル［21］での仮説通り、
その日のデータの欠損や雷雨の影響を受けていた可能性が高いことが明らかになった。
"""
# %%

# グラフのサイズとスタイルを設定
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# 2. Seabornのcountplotを使う
sns.countplot(
    data=day_time,
    y="weather_description",
    order=day_time["weather_description"].value_counts().index,
    hue="weather_description",
    palette="viridis",
    legend=False,
)  # 不要な凡例は非表示にする

# タイトルやラベルを追加
plt.title("Frequency of Weather Descriptions", fontsize=16)
plt.xlabel("Frequency (Count)", fontsize=12)
plt.ylabel("Weather Description", fontsize=12)
plt.tight_layout()

plt.show()

day_time["weather_description"].value_counts()

# %% [markdown]
"""
上記の結果より、セル［22］での疑惑について、

日中期間において、快晴の気象データ以外は、

- shower snow は全て欠損値
- squalls 1つのデータ以外すべて欠損値
- thunderstorm with drizzle 1つのデータ以外すべて欠損値

ということが判明した。  
また、そのほかの気象条件についても、標本サイズが非常に小さかったり、不備のあるものが多いことが分かった。

つまり、セル［21］の棒グラフで示された不自然な事象（平均交通量が冬季気象に影響をほとんど受けないこと）は、標本サイズが
アンバランスなデータの影響を受けている可能性が非常に高い。
この場合、当該棒グラフは偏った平均値を反映していることになるため、あのままの状態では参考にならないし、
同セルで立てた「暖かい時期でも雷雨やスコールの発生しやすい時期はさけるべきである」という仮説も効力を失う。
"""
# %% [markdown]
"""
上記のような不備が発覚したので、改めて正しいデータを使ったセル［21］の棒グラフの改訂版を作っていく。
新しい棒グラフを作成するにおいて、サンプルサイズが十分に大きく、かつ暖かい時期に発生する気象条件に絞って
トップ10の気象項目を特定していく。
"""
# %%
warm_season_weather_count = day_time[month_condition][
    "weather_description"
].value_counts()
display(warm_season_weather_count)

# %% [markdown]
"""
以下の10項目が条件に当てはまる

1. `sky is clear`
2. `scattered clouds`
3. `broken clouds`
4. `mist`
5. `light rain`
6. `overcast clouds`
7. `few clouds`
8. `moderate rain`
9. `light intensity drizzle`
10. `haze`

これより、この10項目について、改めて平均交通量を示す棒グラフを作成する。
"""
# %%
target_weather_list = [
    "sky is clear",
    "mist",
    "broken clouds",
    "overcast clouds",
    "scattered clouds",
    "light rain",
    "few clouds",
    "haze",
    "moderate rain",
    "light intensity drizzle",
]

warm_season_daytime = day_time[month_condition]

# 2. 元のデータフレームから、指定した天候のデータだけを抽出する
filtered_day_time = warm_season_daytime[
    warm_season_daytime["weather_description"].isin(target_weather_list)
]

# 3. 絞り込んだデータを使って、天候ごとの平均交通量を計算し、並べ替える
by_weather_description_selected = (
    filtered_day_time.groupby("weather_description")["traffic_volume"]
    .mean()
    .sort_values()  # 昇順（小さい順）に並べ替え
)

# 4. 絞り込んだデータセットの中で、最大の交通量とその天候名を取得
max_volume = by_weather_description_selected.max()
max_weather = by_weather_description_selected.idxmax()


# 5. グラフの描画
plt.figure(figsize=(12, 8))

# 色のリストを作成（指定した天候リストのインデックスに対してループ）
colors = [
    "red" if weather == max_weather else "#1f77b4"
    for weather in by_weather_description_selected.index
]

# 横向き棒グラフを描画
by_weather_description_selected.plot.barh(color=colors)

# 各棒の右側に交通量の数値を表示する
for index, value in enumerate(by_weather_description_selected):
    font_weight = "bold" if value == max_volume else "normal"
    plt.text(
        value + 30,
        index,
        f"{value:.0f}",
        va="center",
        color="black",
        fontweight=font_weight,
    )


plt.title(
    "Average Traffic Volume for Major Weather Descriptions (Warm Season Daytime)",
    fontsize=16,
)
plt.xlabel("Average Traffic Volume", fontsize=12)
plt.ylabel("Weather Description", fontsize=12)

plt.grid(True, axis="x", linestyle="--", alpha=0.6)
plt.tight_layout(pad=2.0)
plt.show()

# %% [markdown]
"""
４月～１０月（日中）の天候別平均交通量を見ると、全体として差は大きくないものの、
`scattered clouds` のような「曇り寄り」の天候で交通量がやや高く、
`mist` / `haze` のような視界不良に関連する天候でやや低い傾向が見られる。

この差が偶然のブレではなく、統計的に意味のある差（母平均の差）と言えるのかを確かめるため、
以降では `scattered clouds` と `mist` を取り上げて平均交通量の差の検定を行う。
"""

# %% [markdown]
"""
これより、今回行う検定の前提条件を確認していく。
サンプルサイズが十分大きいことから正規分布を仮定し、等分散性は必ずしも仮定できないことからウェルチのt検定を行う。

ただし時系列データを扱う以上、「観測が独立である」という前提がどの程度成立するのかが気になるところである。
自己相関が強いと短期依存性が存在することとなり、自由度が過大評価され、p値が過小評価される可能性があるからだ。

そこで、念のために両天候の平均交通量の分布の形（ヒストグラム）と、自己相関（ACF）を確認する。
"""
# %%

# 日付順にソート（念のため）
warm_season_daytime_sorted = warm_season_daytime.sort_values("date_time").copy()

# 日付列を作成（date_onlyは日単位）
warm_season_daytime_sorted["date_only"] = warm_season_daytime_sorted[
    "date_time"
].dt.date

# 天候ごとに「日次平均交通量」系列を作る（1日1点）
sc_daily = (
    warm_season_daytime_sorted[
        warm_season_daytime_sorted["weather_description"] == "scattered clouds"
    ]
    .groupby("date_only")["traffic_volume"]
    .mean()
)

mst_daily = (
    warm_season_daytime_sorted[
        warm_season_daytime_sorted["weather_description"] == "mist"
    ]
    .groupby("date_only")["traffic_volume"]
    .mean()
)

# indexをdatetime化してソート（ACFを日ラグとして扱うため）
sc_daily.index = pd.to_datetime(sc_daily.index)
mst_daily.index = pd.to_datetime(mst_daily.index)
sc_daily = sc_daily.sort_index()
mst_daily = mst_daily.sort_index()

# 欠けている日を補完（その天候が無い日はNaNのままにするのが自然）
all_days = pd.date_range(
    min(sc_daily.index.min(), mst_daily.index.min()),
    max(sc_daily.index.max(), mst_daily.index.max()),
    freq="D",
)
sc_daily = sc_daily.reindex(all_days)
mst_daily = mst_daily.reindex(all_days)

# ヒストグラム（日次平均交通量の分布）
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(sc_daily.dropna(), bins=50)
plt.title("scattered clouds (daily mean)")
plt.subplot(1, 2, 2)
plt.hist(mst_daily.dropna(), bins=50)
plt.title("mist (daily mean)")
plt.show()

# ACF（自己相関：日次系列としての依存を見る）
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(
    sc_daily.dropna(),
    lags=150,
    title="ACF - scattered clouds (daily mean)",
    ax=axes[0],
    marker=None,
)
plot_acf(
    mst_daily.dropna(),
    lags=150,
    title="ACF - mist (daily mean)",
    ax=axes[1],
    marker=None,
)
plt.show()

# %% [markdown]
"""
ヒストグラムの結果からは、両天候とも左に裾が引っ張られた分布となっているが極端な歪みはなく、
`mist`の方が分布がやや複雑で分散が大きい様子がうかがえるため、
等分散を仮定しないWelchのt検定を選択した判断は妥当であると考えられる。

ACFの結果からも、一見すると持続性があまり強くなく、自己相関は確認できないように見える。
しかし、このACFの横軸ラグは`dropna()`で取得した「観測点数ラグ」であり、連続観測された「日ラグ」として
解釈できるかどうかは、疑問が残る。ACFのラグ1が「翌日」を意味しない場合、連続時系列の依存性を評価することが出来ず、
自己相関（独立性）の確認としては不十分となるため、この点についても確認しておきたい。

そこで、reindex 後の index が日次等間隔であること、`dropna()` 後の観測点数がどの程度連続しているか
（日付の飛びがどの程度あるか）をチェックする関数を定義し、両天候の系列に適用してみる。
"""
# %%


def check_daily_series(name, s):
    # reindex後のindexが日次等間隔か（NaN含む全体でチェック）
    diffs_full = s.index.to_series().diff().dropna()
    is_daily_full = diffs_full.eq(pd.Timedelta(days=1)).all()

    # dropna後に日付が飛び飛びになっていないか（ACFに入れる実データ側）
    s_nonan = s.dropna()
    diffs_nonan = s_nonan.index.to_series().diff().dropna()
    top_diffs = diffs_nonan.value_counts().head(5)

    print(f"\n[{name}]")
    print("Full index is daily spacing (after reindex)?", is_daily_full)
    print("Non-NaN count:", len(s_nonan), "/", len(s))
    print("Top time gaps in non-NaN dates:")
    print(top_diffs)

    # 「lag=1を1日」と読める度合い（非NaN日付の隣接差分が1日である割合）
    if len(diffs_nonan) > 0:
        ratio_1day = diffs_nonan.eq(pd.Timedelta(days=1)).mean()
        print("1-day gap ratio in non-NaN dates:", round(float(ratio_1day), 3))
    else:
        print("Not enough non-NaN points to assess gaps.")


check_daily_series("sc_daily", sc_daily)
check_daily_series("mst_daily", mst_daily)

# %% [markdown]
"""
チェック結果より、index 自体は日次等間隔だが、`dropna()` 後（観測点数ベース）では日付の飛びが無視できない。
実際、非欠損の観測日は `sc_daily` が 621/2188、`mst_daily` が 426/2188 で、
連続した 1 日間隔の割合もそれぞれ 0.618、0.569 に留まっており、独立性の確認用途のACFとして許容範囲であるとは言い難い。

そこで、曜日×時間帯の平均パターンを差し引いた残差系列を用意し、残差ACFで短期依存を確認する。
天候と交通量の関係性に影響を与えうる時間的要因の中でも、短期依存性に特に影響を与えそうなものとして、

- 曜日（週末・平日効果）
- 時間帯（日中のラッシュアワー効果）

が挙げられると判断したためである。
"""
# %% [markdown]
r"""
残差ACFを作成するに当たり、以下のロジックを適用する

- 水準「`weekday`$\times $`hour`の交通量」の群平均（非パラメトリックな推定値）を$\hat{y}_{t}$とみなして
元データの各観測値$y_{t}$から引いて、残差（水準内変動）$e_{t} = y_{t} - \hat{y}_{t}$を作る。
つまり、残差の正体は時間と曜日の時間的要因（規則成分）を取り除いた対象の天候$\times $交通量のデータとなる。

-  残差を時系列順にソートして、その残差系列に対してACFをプロットする

- この残差ACFは、曜日と時間帯による規則的変動を差し引いた後でも短期的な依存（自己相関）
が強く残っていないかを点検するために用いる。これにより、元系列のACFよりも「規則性由来の自己相関」を
切り分けて解釈しやすくなり、t検定で仮定する独立性が大きく損なわれていないかの参考情報になる。
"""
# %%
tmp = warm_season_daytime.copy()
tmp_sorted = tmp.sort_values("date_time")

tmp_sorted["hour"] = tmp_sorted["date_time"].dt.hour
tmp_sorted["weekday"] = tmp_sorted["date_time"].dt.dayofweek  # 0=Mon

# 時間帯×曜日の期待値を引いて残差に変換してから計算
mu = tmp_sorted.groupby(["weekday", "hour"])["traffic_volume"].transform("mean")
tmp_sorted["resid"] = tmp_sorted["traffic_volume"] - mu

# 天候別の残差系列（観測順に並んだ“残差”）を取り出す
sc_resid = tmp_sorted.loc[
    tmp_sorted["weather_description"] == "scattered clouds", "resid"
].dropna()
mst_resid = tmp_sorted.loc[tmp["weather_description"] == "mist", "resid"].dropna()

# ACF（残差に自己相関が強く残ってないか）
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(
    sc_resid,
    lags=300,
    ax=axes[0],
    title="ACF of residuals - scattered clouds",
    marker=None,
)
plot_acf(mst_resid, lags=300, ax=axes[1], title="ACF of residuals - mist", marker=None)
plt.show()

# %% [markdown]
"""
残差ACFの結果、短ラグでは一定の自己相関が残るものの、ラグが進むにつれて減衰し、
長期にわたって強い依存が支配的であるとは言いにくい形状である。
時系列データにおける完全な独立性の仮定は難しいものの、
強い自己相関による検定への悪影響は限定的であると判断できる。

この前提のもと、等分散を仮定しない Welch の t 検定により、
`scattered clouds` と `mist` の平均交通量の差が統計的に有意かどうかを検証する。
"""
# %%

# Pythonの数値リストをRのベクトルに変換
r_scattered = FloatVector(sc_daily.tolist())
r_mist = FloatVector(mst_daily.tolist())

# Welchのt検定を実行
t_test = ro.r["t.test"](r_scattered, r_mist)

"""
Rの生出力をそのまま表示したい場合は、以下のコメントアウトを外して実行
print("\n".join(ro.r("capture.output")(t_test)))
"""

# Rの生出力だと出力結果が長くて見栄えが悪いので、必要な情報のみ取捨選択
t_value = float(t_test.rx2("statistic")[0])
df = float(t_test.rx2("parameter")[0])
p_value = float(t_test.rx2("p.value")[0])
conf_interval = list(t_test.rx2("conf.int"))
means = list(t_test.rx2("estimate"))

# 見やすいフォーマットで出力
print("Welch's t-test: scattered clouds vs mist")
print(f" t(df = {df:.1f}) = {t_value:.3f}, p = {p_value:.3g}")
print(f" 95% CI: [{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]")
print(f" mean(scattered clouds) = {means[0]:.2f}")
print(f" mean(mist) = {means[1]:.2f}")

# %% [markdown]
r"""
上記の結果をまとめると、次のようになる。

$$
\begin{aligned}
 & H_0:\ \mu_{\text{scattered}} - \mu_{\text{mist}} = 0 \\
 & H_1:\ \mu_{\text{scattered}} - \mu_{\text{mist}} \neq 0 \\
 & \text{Test stat: } t = 4.181 \quad (\mathrm{df} = 678.4) \\
 & \text{p-value: } 1.75 \times 10^{-3} \\
 & \text{Decision: 有意水準 } 5\% (\alpha = 0.05) \text{ で } H_0 \text{ を棄却} \\
 & \text{Conclusion: 95\%信頼区間が [105.69, 458.07] と 0 を含まないため、} \\
 & \text{平均交通量に統計的に有意な差が認められる。}
\end{aligned}
$$

検定の結果、p値は 0.05 を下回り、2つの天候における平均交通量の差は統計的に有意であることが確認できた。
ただし、標本サイズが大きいと検出力が高くなるため、差が小さくても有意になりやすい点には留意が必要である。
また、平均値の差は約282台（約5.7%）に留まっており、この差が今回のマーケティング施策において直ちに対策すべきものどうかは、
施策コストや期待できるインプレッション増加率等との兼ね合いで判断すべきである。
"""
# %% [markdown]
"""
これまでの結果を踏まえると、セル［23］の10種類の天候のうち、曇りや小雨のときに多少交通量が多くなる傾向があり、
視界不良に関連する天候（`haze`や`mist`）が起きやすい日は有意差約6%で平均交通量が少なくなる。

よって、広告効果を最大化するためには、可能であれば

- 交通量が比較的多くなる`scattered clouds`の時期に集中投下したい
- 交通量が比較的少なくなる`haze`, `mist`の時期を極力避けたい

なので、これから「特定の天候になりやすい時期はあるのか」を調べていく。
まずは、ACFを作成し、各天候の自己相関の度合から「そもそも周期性があるかどうか」を確認する。

---

※ ここでのACFは、日次カウント系列（日付ごとにその天候が観測された回数）に対して計算する。
セル［26］・［28］で確認した`scattered clouds`と`mist`の **平均交通量** の自己相関とは異なり、
**天候の出現頻度** の自己相関を評価する点に注意。
"""
# %%

# 日付×天候ごとに出現回数を数えて、行=日付 / 列=天候 の日次カウント表に変換
daily_counts = (
    i_94.groupby(["date_only", "weather_description"])
    .size()
    .unstack("weather_description")
    .fillna(0)  # その日その天候の出現が無い（NaN）部分を0回として扱う
)

# 念のために日付をDatetimeIndexに変換してソートして時系列の順序を保証
daily_counts.index = pd.to_datetime(daily_counts.index)
daily_counts = daily_counts.sort_index()

# 欠けている日を埋めて、等間隔の日次系列を保証
all_days = pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq="D")
daily_counts = daily_counts.reindex(all_days, fill_value=0)

acf_weathers = ["scattered clouds", "haze", "mist"]
fig, axes = plt.subplots(1, len(acf_weathers), figsize=(18, 4), sharey=True)
lags = 900

for ax, w in zip(axes, acf_weathers):
    if w not in daily_counts.columns:
        continue
    series = daily_counts[w]
    plot_acf(series, lags=lags, ax=ax, zero=False, marker=None)
    ax.set_title(f"ACF of daily counts: {w}")
    ax.set_xlabel("Lag(days)")

# %% [markdown]
"""
上記の結果を解釈する前に、念のために、天候ごとに抽出したデータが等間隔（例：1日刻み）の時系列として扱えるかを確認する。
ACFの解釈は、等間隔の時系列データであることが前提となるためだ。
"""
# %%

# 日次の等間隔チェック（隣接差分がすべて1日か）
diffs = daily_counts.index.to_series().diff()
print(diffs.value_counts().head(5))
print("All daily spacing?", diffs.dropna().eq(pd.Timedelta(days=1)).all())

# 1行=1日になっているか（date range と行数の整合）
n_days = len(daily_counts)
days_span = (daily_counts.index.max() - daily_counts.index.min()).days
print("n_days =", n_days)
print("date range:", daily_counts.index.min(), "->", daily_counts.index.max())
print("days span:", days_span)
print("n_days == days_span + 1 ?", n_days == days_span + 1)

# %% [markdown]
"""
確認の結果、日付は1日刻みで欠けなく並んでおり（`All daily spacing = True`）、
daily_counts は日次の等間隔系列として扱える。
したがって、ACFの lag=1 は「1日遅れ」として解釈できる。
"""
# %% [markdown]
r"""
改めて、ACFの出力の結果、

- どの天候も、自己相関の度合いが信頼区間の内側に留まっており、この結果からは明確な周期性が存在するとは言えない
- `scattered clouds`は、「天候の持続性」に長期トレンドがあると捉えられる
- `haze`と`mist`は、持続性が`scattered clouds`に比べると低い

という解釈ができそうだ。  
したがって、ACFだけでは周期性の有無を断定できそうにない。  
"""
# %% [markdown]
r"""
次に、時間帯$\times $天候のヒートマップで、具体的に「いつその天気になりやすいか」を相対頻度で確認し、周期性の有無を検証する。
これまでの分析結果を踏まると、4~10月の日中（6〜18時）に絞って分析するのが妥当であると考えられる。
よって、まずは暖かい時期の日中に絞ってデータを抽出し、
その中で各天候が観測された回数を月ごと・時間帯ごとに集計し、ヒートマップで可視化する。
"""
# %%

# 暖かい時期 & 日中 （例: 4〜10月 / 6〜18時）に絞ってデータを抽出
target_weathers = ["scattered clouds", "haze", "mist"]


def plot_weather_heatmap(weather, normalize=True):
    df = warm_season_daytime[
        warm_season_daytime["weather_description"] == weather
    ].copy()

    # その天気が観測された回数
    counts = (
        df.groupby(["month", "hour"])
        .size()
        .unstack("hour")
        .reindex(index=range(4, 11), columns=range(6, 19))
        .fillna(0)
    )

    if normalize:
        # ベース：全天気での観測回数
        base = (
            warm_season_daytime.groupby(["month", "hour"])
            .size()
            .unstack("hour")
            .reindex(index=range(4, 11), columns=range(6, 19))
            .fillna(0)
        )
        data = counts / base.replace(0, np.nan)
        cbar_label = "Relative frequency"
    else:
        data = counts
        cbar_label = "Count"

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        data,
        cmap="viridis",
        linewidths=0.2,
        linecolor="gray",
        cbar_kws={"label": cbar_label},
    )
    plt.title(f"'{weather}' frequency (Apr–Oct, daytime)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.show()


for w in target_weathers:
    plot_weather_heatmap(w, normalize=True)

# %% [markdown]
"""
結果として、以下のことがわかった。

- `scattered clouds`は、５～8月の夕方にかけて発生頻度が高く、特に7月の14〜17時台にピークがある
- `haze`は明確なトレンドはないが、4月と9月の昼頃に発生頻度がやや高い
- `mist`は時期を問わず朝に発生頻度が高く、昼以降は減少する傾向がある。8月の6〜8時台にピークがある

いずれの天候も相対頻度のスケールが低く、明確な周期性は見られないが、
`scattered clouds`の発生頻度の分析の結果は、これまでの「時間的要因と交通量の分析」で示された洞察と整合的であり、
「暖かい季節の夕方に交通量が多くなる」という傾向を補強するものとなった。

また、
`scattered cloud`が発生しやすい時期と、`haze`及び`mist`が発生しやすい時期が被っていないことからも、
今回のマーケティング施策はこれまでの分析でも示唆された通り、夏の夕方に優先的に予算を割り当てるのが妥当であると考えられる。

<hr>
【補足】
この傾向が統計的に有意かどうかを検証するために、天候と時間帯の独立性をカイ二乗検定で調べることもできるが、
サンプルサイズが大きいことから検出力が高くなりやすいことと、ヒートマップからおおよその相対頻度を確認できることから、
EDAの段階ではここまでとする。
"""
# %% [markdown]
"""
#### ロバスト性の確認

最後に、日中データ全体における月別の交通量統計量を確認し、
これまでの分析で示された「暖かい季節に交通量が多くなる」という洞察が全体データにおいても成り立つかどうかを確認する。

これまでの分析では、「最も交通量が多くなると期待できる条件」を明らかにすることを目的としていたため、平均値について
のみ注目していた。その結果、4〜10月（特に４月、10月）に交通量が多くなるという洞察が得られたわけだが、これが果たして、
中央値についても同じような傾向が見られるのかどうかを確認する。

中央値は、外れ値の影響を受けにくい指標であるため、もし中央値についても同様の傾向が見られれば、
「暖かい季節に交通量が多くなる」という洞察がより強固なものとなるだけでなく、ビルボードの広告効果を最大化するための
戦略立案においても有用な情報となる。

よって、これから日中データ全体における月別の交通量の平均値と中央値を計算し、比較していく。
この分析の目的は、最終確認として保守的な視点からこれまでの洞察を検証することであるため、
99パーセンタイルでトリミングした場合の結果も同時に示すこととする。
"""
# %%
daytime_clean = day_time.dropna(subset=["traffic_volume"]).copy()
stats_raw_full = daytime_clean.groupby("month")["traffic_volume"].agg(
    count="size", mean="mean", median="median"
)
# 上位1%を外してトリミング（全日中の99パーセンタイルを閾にする）
p99_full = daytime_clean["traffic_volume"].quantile(0.99)
day_trim_full = daytime_clean[daytime_clean["traffic_volume"] <= p99_full].copy()
# トリミング後の月別集計
stats_trim_full = day_trim_full.groupby("month")["traffic_volume"].agg(
    count="size", mean="mean", median="median"
)
# 比較用に結合（列名に接頭辞を付ける）
compare_full = stats_raw_full.add_prefix("raw_").join(
    stats_trim_full.add_prefix("trim_")
)
# 表示（trim_median の降順で並べ替えて表示）
display(compare_full.sort_values("trim_median", ascending=False))

# %% [markdown]
"""
上記の結果と、これまでの分析結果を比較すると、

- セル［10］で示された平均値の分析では、
 1. 4月
 2. 10月
 3. 次点で5月６月８月が続く

という順番で交通量が多いことがわかったが、

- 今回の中央値の分析では、
 1. 10月
 2. 4月
 3. 次点で8月６月５月が続く

という順番で交通量が多いことがわかった。

従って、中央値に基づく分析でも、暖かい季節（4〜10月）に交通量が多くなる傾向が確認できた。
こうなると、4〜10月を優先ターゲットにすることは前提となり、そのうえで、何月に施策の予算を多く割り当てるかは、
保守的な姿勢（中央値ベース）ととるか、積極的な姿勢（期待値ベース）ととるかによって変わってくるため、
マーケティング部門の方針次第となるだろう。
"""
# %% [markdown]
r"""
<a id="conclusion"></a>
## まとめ

本プロジェクトでは、I‑94州間高速道路（西行き）の交通量データを用いて、
デジタルビルボードの広告効果を高めるために、

- 「どのような時間帯・曜日・季節に交通量が多くなるのか」
- 「どのような気象条件のときに交通量が増えたり減ったりするのか」

を探索的データ分析（EDA）によって調べた。

### 1. 時間的要因から見えたこと

- **季節**  
  - 日中データに限定した分析では、**4〜10月の暖かい季節に交通量が多く、冬季に減少する**周期性が確認できた。
- **曜日**  
  - 週末よりも平日の交通量が多く、**平日の中では、特に木・水・金の順に交通量が多い**傾向がある。
- **時間帯**  
  - 平日に限ると、**朝6〜8時台と夕方15時代～17時台にラッシュアワーがあり、特に16時頃が最大**となる。

また、時間軸の分析の途中で、

- 2016年7月の一部期間で交通量が大きく落ち込む
- 2014年8月7日〜2015年6月12日にかけて交通量がほぼ変動しない

といった「不自然な推移」を発見したが、  
どちらもデータの異常や欠損の可能性が高いことを確認した。


### 2. 気象条件から見えたこと

- 量的変数としての `temp`（ケルビン温度）は、`traffic_volume` との相関が弱く、**気温だけから交通量変動を説明するのは難しい**
 ことがわかった。

- カテゴリ変数としての `weather_main` / `weather_description` を用いた分析では、
 快晴よりも、**曇りや小雨など「やや悪天候」のときに交通量が多い**傾向 が見られた。

- 一方で、`haze` や `mist` など**視界不良となる天候では平均交通量が比較的低い**

これらの結果から、ミネソタ州の湿潤大陸性気候により、雲がかかってくると現地住民は雨を警戒して移動手段を車にするかもしれない
こと、冬季の交通量が減少する原因についても、同様に冬場の厳しい気象に起因するものかもしれないという考察を行った。

また、暖かい季節（4〜10月）の日中データに絞り、  
十分な標本サイズをもつ主要な気象条件10種類に限定して再集計したところ、

- 全体としてはそこまで極端な差はないものの、**scattered clouds のときに最も交通量が多く、hazeやfog のときにやや少ない**

というパターンが棒グラフの頻度分析により視認できた。

このパターンの統計的有意性を検証するため、`scattered clouds` と `mist` の日中交通量について、R（rpy2）による
Welch の t 検定を行った結果、

- P値が$ 4.181 \times  10^{-3} $と十分に小さい
- 母平均の差の95%信頼区間が約 105～458 台／時（約 2〜10%差）

であることから、**統計的には有意な差がある**と判断した。

ただし、標本サイズ・自由度が高いため、検出力が高くなってしまった可能性を考慮すべきであり、
**実務上どこまで重要な差とみなすかは、広告コストとのバランスを見ながら判断すべき**である。


また、`scattered clouds`・`haze`・`mist`の3つの天候について、ACFとヒートマップを用いて周期性の有無を調査した結果、
いずれの天候も明確な周期性は見られなかったが、

- `scattered clouds`は5〜8月の夕方にかけて発生頻度が高く、特に**7月の14〜17時台にピークがある**
- `haze`は4月の昼頃に発生頻度がやや高い
- `mist`は時期を問わず朝に発生頻度が高く、昼以降は減少する傾向がある。8月の6〜8時台にピークがある

という傾向が見られた。

特に、`scattered clouds`の発生頻度の分析の結果は、これまでの「時間的要因と交通量の分析」で示された洞察と整合的であり、
「暖かい季節の夕方に交通量が多くなる」という傾向を補強するものとなった。

そして、最終確認としてロバストネスチェックを行った。
日中データ全体における月別の交通量の中央値を計算し、平均値ベースの分析結果と比較したところ、以下のような順位の違いが見られた。

- 平均値ベース（期待値重視）： 1位 4月、2位 10月、次点で5・6・8月が続く
- 中央値ベース（外れ値の影響を除外）： 1位 10月、2位 4月、次点で8・6・5月が続く

順位に若干の変動はあるものの、どちらの指標でも 暖かい季節（4〜10月）に交通量が多くなる という大局的な傾向は一致している。

したがって、4〜10月を優先ターゲットにすることは前提 となるだろう。
そのうえで、具体的に何月に予算を重点配分するかは、保守的な姿勢（中央値ベース）をとるか、
積極的な姿勢（期待値ベース）をとるかによって変わってくるため、マーケティング部門の方針次第である。

---

### 3. 広告戦略の観点での示唆

広告配信設計としては、「4〜10月の暖かい季節」を狙うことは前提条件にしたい。ミネソタ州の厳しい冬季を避けるためである。
この際、その中でもどの月に重点的に配信するかは、広告予算とリスク許容度に応じて決定すべきである。

また、以下の点を考慮することで、広告効果を最大化できると考えられる。
優先度の高い順に、

1. 「 **週末よりも平日** に重点的に配信する」
2. 「平日の中でも特に月曜日の実施に割くコストを減らし、 **木曜日・水曜日・金曜日** の順番で優先的に予算を割く」
3. 「1日の中でも特に **夕方15〜18時台** に集中的に配信する」

そのうえで、予算とスケジュールに余裕がある場合は、
「曇りや小雨のときに配信を強化し、視界不良となる天候（haze, mist）のときは配信を控える」など、天候APIを活用した戦略も
検討の余地がある。
"""
