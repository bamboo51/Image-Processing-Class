// Example program showing how to import and use the pgm module.
// Run with: go run ./example
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"example.com/m/pgm"
	"example.com/m/utils"
)

func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("-----------------------------------------------------")
	fmt.Println("  モノクロ階調画像入力ルーチン")
	fmt.Println("-----------------------------------------------------")
	fmt.Println("ファイル形式は pgm, バイナリ形式とします．")
	fmt.Print("入力ファイル名 (*.pgm) : ")
	inFile, _ := reader.ReadString('\n')
	inFile = strings.TrimSpace(inFile)

	img, err := pgm.Load(inFile)
	if err != nil {
		fmt.Println("エラー:", err)
		os.Exit(1)
	}
	fmt.Printf("横の画素数 = %d, 縦の画素数 = %d\n", img.Width, img.Height)
	fmt.Println("データは正しく読み込まれました．")
	fmt.Println("-----------------------------------------------------")

	hist := utils.MakeHistogramImage(img)

	fmt.Println("-----------------------------------------------------")
	fmt.Println("  モノクロ階調画像（pgm形式）出力ルーチン")
	fmt.Println("-----------------------------------------------------")
	fmt.Print("出力ファイル名 (*.pgm) : ")
	outFile, _ := reader.ReadString('\n')
	outFile = strings.TrimSpace(outFile)

	if err := pgm.Save(outFile, hist); err != nil {
		fmt.Println("エラー:", err)
		os.Exit(1)
	}
	fmt.Println("データは正しく出力されました．")
	fmt.Println("-----------------------------------------------------")

	fmt.Println("-----------------------------------------------------")
	fmt.Println("  モノクロ階調画像（pgm形式）出力ルーチン")
	fmt.Println("-----------------------------------------------------")
	fmt.Print("出力ファイル名 (*.pgm) : ")
	sobelOutfile, _ := reader.ReadString('\n')
	sobelOutfile = strings.TrimSpace(sobelOutfile)

	out, err := utils.SobelFiltering(img)
	if err != nil {
		fmt.Println("エラー:", err)
		os.Exit(1)
	}
	if err := pgm.Save(sobelOutfile, out); err != nil {
		fmt.Println("エラー:", err)
		os.Exit(1)
	}
	fmt.Println("データは正しく出力されました．")
	fmt.Println("-----------------------------------------------------")
}
