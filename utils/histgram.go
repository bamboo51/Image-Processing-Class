package utils

import (
	"fmt"

	"example.com/m/pgm"
)

const imageSize = pgm.MaxBrightness + 1

func MakeHistogramImage(img *pgm.Image) *pgm.Image {
	histogram := make([]int, imageSize)
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			histogram[img.Pixels[y][x]]++
		}
	}

	maxFrequency := histogram[0]
	for i := 1; i < imageSize; i++ {
		if histogram[i] > maxFrequency {
			maxFrequency = histogram[i]
		}
	}
	fmt.Printf("maxFrequency: %d\n", maxFrequency)

	out := &pgm.Image{
		Width:  imageSize,
		Height: imageSize,
		Pixels: make([][]uint8, imageSize),
	}

	for y := range out.Pixels {
		out.Pixels[y] = make([]uint8, imageSize)
	}

	if maxFrequency == 0 {
		return out
	}

	for i := 0; i < imageSize; i++ {
		barHeight := int(float64(pgm.MaxBrightness) / float64(maxFrequency) * float64(histogram[i]))
		for j := 0; j < barHeight; j++ {
			out.Pixels[imageSize-1-j][i] = pgm.MaxBrightness
		}
	}
	return out
}
