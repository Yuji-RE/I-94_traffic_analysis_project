## 時間データの定義一貫性の確認

- 「トップ10 天候」選定 → 実際の棒グラフ作成のフィルタ漏れ
  - 使っているデータ:
    - `warm_season_weather_count = day_time[month_condition]["weather_description"].value_counts()` で warm-season 順位を作成（OK）
    - しかしその後 filtered_day_time = day_time[day_time["weather_description"].isin(target_weather_list)] を作成する際に month_condition（4–10月）が再適用されていない → 全年の day_time を使って平均を出している
  - 問題: 「暖かい時期のトップ10」を根拠にしておきながら、実際の平均やプロットは全年（日中のみ）を使っているため結果がずれる可能性あり。
  - 最小修正案（1行）:
    - filtered_day_time = day_time[day_time["weather_description"].isin(target_weather_list) & day_time["date_time"].dt.month.between(4,10)]

- Welch の t 検定（scattered vs haze）
  - 使っているデータ: sc / hz を day_time から抽出しているが、month フィルタ（4–10月）を明示していない
  - 問題: 検定は「暖かい時期の差」を見たい意図に見えるため、暖かい時期に限定していないと結論が混ざる。
  - 最小修正案:
    - scattered = day_time.loc[(day_time["weather_description"]=="scattered clouds") & (day_time["date_time"].dt.month.between(4,10)), "traffic_volume"]
    - haze      = day_time.loc[(day_time["weather_description"]=="haze") & (day_time["date_time"].dt.month.between(4,10)), "traffic_volume"]

- plot_traffic_pattern の一連プロット（hour/day/month）
  - 使っているデータ: sorted_weather_list を warm-selected から作っているが、ループ内で weather_specific_data = day_time[day_time["weather_description"] == weather].copy() としており month フィルタがない
  - 問題: プロットは「暖かい時期に限定したい」意図に見えるが、実際は全年の日中データがプロットされる
  - 修正案: weather_specific_data = day_time[(day_time["weather_description"] == weather) & (day_time["date_time"].dt.month.between(4,10))].copy()

- ACF（日別カウントの自己相関）
  - 使っているデータ: daily_counts = i_94.groupby(["date_only","weather_description"]).size().unstack(...).fillna(0)
  - 問題: ACF の解釈文で「4–10月の日数ベース」と言っている一方、集計自体は i_94 全体（全月）を使っている → ACF の結果と文脈がずれる。
  - 修正案: daily_counts を warm-season のみで作る:
    - warm = i_94[i_94["date_time"].dt.month.between(4,10)]
    - daily_counts = warm.groupby(["date_only","weather_description"]).size()...

細かい不一致（注意しておいた方が良い点）
- day_time の時間定義の微妙な不一致:
  - 初期定義: day_time = i_94[(i_94["hour"] >= 6) & (i_94["hour"] <= 18)]
  - heatmap の warm_daytime: hour.between(7,18)
  - 効果: 6時台のデータが heatmap に含まれない可能性がある（微小な違いだが一貫性のため合わせた方がよい）。
  - 修正案: どちらかに揃える（例：全て hour.between(6,18) に統一）。
