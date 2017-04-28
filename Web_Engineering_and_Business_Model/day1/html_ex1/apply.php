<?php
$sirname="";
$firstname="";
//var_dump($_SERVER['REQUEST_METHOD']);
if($_SERVER['REQUEST_METHOD']=='POST'){
	// var_dump($_POST);
	$sirname=$_POST['sirname'];
	$firstname=$_POST['firstname'];
	// $err =false;
}
?>
<!DOCTYPE html>
<html lang="ja">
	<meta charset="UTF-8" />
<head>
	<title>Webプログラミング入門</title>
</head>
	<link rel="stylesheet" type="text/css" href="css/cssreset-min.css" />
	<link rel="stylesheet" type="text/css" href="css/style.css" />
	<link rel="stylesheet" type="text/css" href="css/form.css" />
<body>
	<div id="container">
		<div class="section">
			<div id="top">
				<h1>Hello
					<?php
					if(empty($sirname)){
						echo 'ヤギ ';
					}else{
						echo $sirname;
					}

					if(empty($firstname)){
						echo 'です';
					}else{
						echo $firstname;
					}
					?>
				</h1>
			</div>
		</div>
		<div class="section">
			<div id="photo">
				<img src="img/yagi.jpg">
			</div>
		</div>
		<div class="section">
			<div id="apply-form">
				<form	action="" method="POST" id="ss-form" target="_self">
					<p>名前急募<p>
						<div class="ss-form-question">
							<div class="ss-q-title">名字</div>
							<input type="text" class="ss-q-short" name="sirname">
							<div class="ss-q-title">名前</div>
							<input type="text" class="ss-q-short" name="firstname">
							<input type="submit" id="ss-submit" name="submit" value="送信" >
						</div>
				</form>
			</div>
		</div>
		<div class="section" id="explanation">
			<div class="topic" id="news">
						<p>[プロフィール]</p>
						<ul>
							<li>属性：ウシ科ヤギ属</li>
							<li>分類：家畜</li>
							<li>特徴：<a href="https://en.wikipedia.org/wiki/Goat" target="_blank" >こちら</a>を参照ください</li>
							<li>名前：まだない</li>
						</ul>
			</div>
		</div>
		<div class="section" id="explanation">
			<div class="topic" id="news">
						<p>[家畜としてのヤギ]</p>
						<p>
							ヤギは家畜として古くから飼育され、用途により乳用種、毛用種、肉用種、乳肉兼用種などに分化し、その品種は数百種類に及ぶ。
							ヤギは粗食によく耐え、険しい地形も苦としない。そのような強靭な性質から、山岳部や乾燥地帯で生活する人々にとって貴重な家畜となっている。
							ヤギの乳質はウシに近く、乳量はヒツジよりも多い。明治以降、日本でも数多くのヤギが飼われ、「貧農の乳牛」とも呼ばれたが、高度経済成長期を境として減少傾向にある。
							しかし、近年ではヤギの愛らしさ、粗放的飼育に耐えうる点等が再評価されつつある。
							これを受けて、ヤギ愛好者・生産者・研究者が一堂に会する「全国山羊サミット」が年に1回、日本国内で毎年開催場所を変えて開催されており、年々盛況になっている。
						</p>
			</div>
		</div>
	</div>
</body>
</html>
