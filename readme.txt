実行環境・ライブラリ：
	使用機器：NVIDIA AGX Xavier (Jetpack 4.4.0) (OS: Ubuntu 18.04 LTS)
	Python: 3.6.9	 
	ライブラリ：
		・opencv 3.4.11
		・cupy 8.3.0
		・numpy 1.20.0
		・matplotlib 3.3.4

実行方法：
githubを利用できる場合:
	git clone https://github.com/kohtaro246/robot_intel.git
	を実行するとデータ準備の必要はありません。3月下旬までは公開している予定です。

データ準備（githubを使用しない場合）：
	https://www.cs.toronto.edu/~kriz/cifar.html
	1. このウェブサイトでCIFAR-10 python versionをダウンロードし、プログラムファイルと同じディレクトリで展開してください。
	2. 展開するとcifar-10-pythonというディレクトリができるので、このディレクトリに移動し、以下の５つの空ディレクトリを作成してください。
	   noise_5
	   noise_10
	   noise_15
	   noise_20
	   noise_25
	3. プログラムの入っているディレクトリに戻り、"python3 create_noise.py"を実行してください。自動的にデータが生成されます。
学習プログラムの実行：
	"python3 kadai_?_?_?_?.py"を実行する。基本プログラムのファイル名は"kadai_0_0.py"。ファイル名は以下の規則に従う。
	層の数
		kadai_0_0_[層数(_ニューロン数)].py
	学ばせるデータに含まれるノイズの割合
		kadai_0_[ノイズ割合].py
	Weight decay
		kadai_[weight decay小数点省く]_0

	実行すると、パラメータを格納したpickleファイル、精度とcostのグラフ、最終的なcostの値が自動生成される。(生成されるファイルはすでにprogram.zipに入っている)	
評価プログラムの実行：
	1. 407行目のpred = predict(test_set2, params, 143)を実行すると、204行目で開いたパラメータを用いて219行目で指定したテストセットの143番目のデータを分類し、写真を表示します。
	2. 409行目のprob, accu = evaluate(dataset, params)を実行すると、204行目で開いたパラメータを用いて223行目で指定したテストセットの汎化性能をaccuに代入します。
	3. 412行目のnoise_depend(test_sets, pickle_list)を実行すると、考察で用いたノイズ耐性に関するグラフがrobustnessに生成されます（生成されるファイルはすでにprogram.zipに入っている）



補足：もし付録で紹介したプログラムを実行する場合は
git clone https://github.com/kohtaro246/smart_fridge_project.git
をした上で、plate_recognizer.pyの最終行をコメントインして学科PC環境で実行してください。
学習プログラムはplate_learn5.pyです。
