<?php

shell_exec("rm ./upload/*");
shell_exec("rm ./out/*");

$flag = 0;
$allowedExts = array("gif", "jpeg", "jpg", "png" , "pgm", "ppm");
$extension = end(explode(".", $_FILES["file"]["name"]));
if ((($_FILES["file"]["type"] == "image/gif")
			|| ($_FILES["file"]["type"] == "image/jpeg")
			|| ($_FILES["file"]["type"] == "image/jpg")
			|| ($_FILES["file"]["type"] == "image/png"))
		|| in_array($extension, $allowedExts))
{
	if ($_FILES["file"]["error"] > 0)
	{
		echo "Return Code: " . $_FILES["file"]["error"] . "<br>";
	}
	else
	{
	/*	echo "Upload: " . $_FILES["file1"]["name"] . "<br>";
		echo "Type: " . $_FILES["file1"]["type"] . "<br>";
		echo "Size: " . ($_FILES["file1"]["size"] / 1024) . " kB<br>";
		echo "Temp file: " . $_FILES["file1"]["tmp_name"] . "<br>";
*/
		if (file_exists("/var/www/ass4/upload/" . $_FILES["file"]["name"]))
		{
			echo $_FILES["file"]["name"] . " already exists. ";
		}
		else
		{
			move_uploaded_file($_FILES["file"]["tmp_name"],
					"/var/www/ass4/upload/" . $_FILES["file"]["name"]);
//			echo "Stored in: " . "upload/" . $_FILES["file1"]["name"] . "<br>";
		}
	}
}
else
{
	$flag = 1;
	echo "Invalid file";
}

if ($flag == 0){

//	system("firefox http://run.imacros.net/?m=the_macro.iim 2>&1");
//	system("firefox http://run.imacros.net/?m=the_macro.iim >/dev/null 2>&1");

	$file = "upload/".$_FILES['file']['name'];

	echo "File Upload Successful" . "<br>";

/*	echo $desc . "<br>";
	echo $_POST['Extractor']. "<br>";
	echo $_POST['Matcher']. "<br>";
	echo $_FILES['file1']['name']. "<br>";
	echo $_FILES['file2']['name']. "<br>";*/

	$fname = './a.out 0 1000_30_30.xml upload';
	$out = shell_exec($fname);

//	echo $file . '<br>';
	
	echo '<html><body>';
	echo '<p><img src="'.$file.'" alt="1"></p>';
	echo '<table width="1500" border="10" align="left" cellpadding="5" cellspacing="10">
		
		<tr><td>
		<img src="out/1.png" alt="1">
		</td>

		<td>
		<img src="out/2.png" alt="2">
		</td>

		<td>
		<img src="out/3.png" alt="3">
		</td>

		<td>
		<img src="out/4.png" alt="4">
		</td></tr>

		<tr><td>
		<img src="out/5.png" alt="5">
		</td>

		<td>
		<img src="out/6.png" alt="6">
		</td>

		<td>
		<img src="out/7.png" alt="7">
		</td>

		<td>
		<img src="out/8.png" alt="8">
		</td></tr>

		</table></body></html>';
}
?>
