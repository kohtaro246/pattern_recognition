実行環境・ライブラリ：
	使用機器：自作ｐｃ（GeForce RTX 3070を用いて学習） (OS: Ubuntu 20.04 LTS)
	Python: 3.8.5
	ライブラリ：
		・opencv 3.4.11
		・cupy-cuda101 8.4.0
		・numpy 1.20.0
		・matplotlib 3.3.4
		・tqdm 4.56.0

実行方法：
githubを利用できる場合:
	git clone https://github.com/kohtaro246/pattern_recognition.git
	を実行するとデータ準備の必要はありません。3月下旬までは公開している予定です。

データ準備（githubを使用しない場合）：
	https://www.cs.toronto.edu/~kriz/cifar.html
	1. このウェブサイトでCIFAR-10 python versionをダウンロードし、プログラムファイルと同じディレクトリで展開してください。

学習プログラムの実行：
バッチ学習："python3 kadai_gradient.py"を実行してください。
ミニバッチ学習："python3 kadai.py"の以下の行をコメントインして実行してください。
	学習
		680-687行目
	交差検証法
		690-692行目
	赤池情報量基準
		695行目
	特定の画像に関して推論をする（学習済みのパラメータをpickleファイルに保存してあるので学習をする必要はありません）
		698行目
		



