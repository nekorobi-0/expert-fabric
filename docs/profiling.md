# 処理時間プロファイリング

メイン側のC++プロセスにPrometheus exporterを同梱する。YAMLの`profiling.prometheus`で有効化と待受アドレスを指定し、Prometheusは`/metrics`をscrapeする。実装済みの工程は`command_roundtrip`、`weight_load`、`expert_queue`、`expert_compute`、`ngram_lookup`。ローカル実行ではメインから、RoCE v2のworker側の処理時間はworker自身のexporterから取得できる。起動用設定は [ローカルCPU用YAML](../configs/local-cpu.yml) を参照。

以下はデコーダ統合時の計測設計。

| メトリクス | 計測内容 |
| --- | --- |
| `expert_fabric_stage_duration_seconds` | ヒストグラム。メイン側の投入準備・ターゲット検証、DSparkとexpertの往復、worker側のキュー待ち・演算、ホストとGPU間のコピー、重みロード |
| `expert_fabric_request_duration_seconds` | ヒストグラム。リクエスト受付から最終token出力まで |
| `expert_fabric_inter_token_gap_seconds` | ヒストグラム。各リクエストで実際に出力した連続token間の間隔。DSparkのラウンド時間は工程側で別に記録する |
| `expert_fabric_tokens_total` | カウンタ。生成候補・検証対象・採択候補・確定出力のtoken数 |

長文の比較ではDSparkのSSDロードとdraft状態の再構成、n-gram探索を工程として記録する。候補・検証・採択のカウンタは候補の由来を`dspark`と`ngram`で区別する。

CPU/GPU並行実行の比較では、メイン側がexpert出力を必要とした時点から集約を再開できるまでの待ち時間を記録する。expert往復全体と区別し、計算によって隠せた時間を比較する。

時間は秒に揃える。工程、通信経路（`local`、`upi`、`roce_v2`）、実行先の種別（`cpu`、`gpu`）は必要なメトリクスだけに固定ラベルで付ける。expert IDは専用のカウンタに限る。[Prometheusの命名規約](https://prometheus.io/docs/practices/naming/) · [ヒストグラム](https://prometheus.io/docs/practices/histograms/)

YAMLの`profiling.per_expert_counters`を有効にすると、`expert_fabric_expert_invocations_total{expert_id=...}`と`expert_fabric_expert_tokens_total{expert_id=...}`を公開する。1回の成功したexpert実行に8 tokenをまとめた場合は、前者に1、後者に8を加える。実行経路はロード時に取得したカウンタを使用し、緩和順序のアトミック加算で記録する。[Prometheusの計測コストに関する指針](https://prometheus.io/docs/practices/instrumentation/#inner-loops)

往復時間はメイン側、キュー待ちと演算時間は処理したworker側の単調時計で計測する。workerは経過時間を既存の完了応答に載せ、メイン側で集約する。GPUの演算とコピーを細かく測る場合はYAMLで`profiling.gpu_timing`を有効にし、HIPイベントの経過時間を完了後に回収する。測定のためにGPU全体を同期させない。往復とworker側の計測は範囲が重なるため、合計せず別々に見る。

`/metrics`は集約済みの値を返す。比較時は同じヒストグラム境界と同じ入力条件で、工程のp50/p95/p99、出力間隔のp99、確定token/sを見る。
