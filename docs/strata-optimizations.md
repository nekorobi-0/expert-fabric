# Strataから取り込む高速化

参照したのはQwen系MoEをCPUとGPUで動かす[MoEエンジン版Strata](https://github.com/dhoard/strata/blob/main/docs/DETAILS.md)と、その[Strata-GLM AMD版のdecode実装記録](https://github.com/OpenMOSE/Strata-GLM-AMD/blob/main/docs/GLM5_NEXT.md)。KV転送の項目では同名の[長文KVキャッシュ論文](https://arxiv.org/html/2508.18572v1#S4.SS2)を参照した。以下はExpert Fabric向けの設計と検証順。

1. **CPU expertの常駐先と実行コアを揃える。** `LOAD`時に重みをworkerのNUMAノードへ配置し、固定ワーカープールをそのノードのコアに割り付ける。通信の進行役にもコアを残す。複数ソケットへ広げる場合は、まずexpert単位で分配して、重みを読むためのソケット間転送を減らす。物理コアを基準にした`cpu_workers: auto`と固定ワーカー数を比較する。[StrataのNUMA設計](https://github.com/dhoard/strata/blob/main/docs/DETAILS.md#two-sockets-and-numa)
2. **CPU expertへの投入を早め、集約の直前で待つ。** GPUルーティング後にhidden stateを宛先ごとの事前確保バッファへ詰め、転送ストリームで引き渡す。CPUは必要な入力が揃った時点でexpertを実行し、メインのGPUは独立な共有expertやGPU側に配置されたexpertがあれば並行して処理する。結果の可視化を確認したあと、重み付き集約だけがその結果を待つ。入力と出力のバッファは完了まで再利用しない。[Strata-GLM AMDのCPU/GPU並行実行](https://github.com/OpenMOSE/Strata-GLM-AMD/blob/main/docs/GLM5_NEXT.md)
3. **n-gram先読みは実測で有効化する。** 候補のヒット率だけで判断せず、採択されたtoken数と、検索・検証・CPU expert待ちを含む時間を見る。直近に通常decodeを観測した区間と比べて確定tokenあたりの時間が改善しない場合は、候補長を抑えるか一時停止する。停止中も少量の試行で再評価する。過去リクエスト由来のRAMキャッシュは既存の共有範囲と上限を使う。[Strataのprompt lookup](https://github.com/dhoard/strata/blob/main/docs/DETAILS.md#how-it-works)
4. **長文prefillはチャンクと転送量を一緒に調整する。** 同じチャンクのtokenをexpertごとにまとめ、宛先ごとの事前確保リングでhidden stateと結果を運ぶ。チャンク長は固定値から測り、`auto`では作業用メモリ、転送バイト数、CPU expertの処理時間で決める。長文の再開でKV復元が支配的になった段階では、細かなKVページをまとめてRAMから転送し、次の層のページを計算と重ねる。GPUでのgather/scatterはコピーエンジンによる一括転送と実測で比較する。[Strataのprefill設計](https://github.com/dhoard/strata/blob/main/docs/DETAILS.md#how-it-works) · [Strata論文のKV転送](https://arxiv.org/html/2508.18572v1#S4.SS2)

全expertの公開世代は実行先メモリに常駐させる。GPUのホットexpertキャッシュ、重みのSSDページング、CUDA固有の計測値は採用対象に含めない。Strataが資料で区別しているSSD上のモデル用n-gramテーブルと、生成を先読みするprompt lookupのうち、取り込むのはprompt lookupの採算判定。

最初の比較では配置と入力を揃えて、NUMA固定と遅い段階での待ち合わせを個別に試す。decodeは出力間隔のp99・確定token/s・集約直前の待ち時間、prefillはTTFT・転送量、先読みは候補と採択token数を記録する。MI50、RX 7900 XTX、UPI、RoCE v2で条件を分けて測る。
