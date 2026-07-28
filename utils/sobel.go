package utils

import (
	"fmt"
	"math"

	"example.com/m/pgm"
)

func SobelFiltering(image *pgm.Image) (*pgm.Image, error) {
	weight := [3][3]int{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	}
	const div_const = 1.0

	height, width := image.Height, image.Width
	raw := make([][]int, height)
	for y := range raw {
		raw[y] = make([]int, width)
	}

	fmt.Print("Apply filter to original image")
	min := math.MaxInt
	max := math.MinInt

	for y := 1; y < image.Height-1; y++ {
		for x := 1; x < image.Width-1; x++ {
			pixel_value := 0
			for i := -1; i < 2; i++ {
				for j := -1; j < 2; j++ {
					pixel_value += weight[i+1][j+1] * int(image.Pixels[y+i][x+j])
				}
			}
			pixel_value = pixel_value / int(div_const)
			if pixel_value < min {
				min = pixel_value
			}
			if pixel_value > max {
				max = pixel_value
			}
			raw[y][x] = pixel_value
		}
	}

	if max-min == 0 {
		return nil, fmt.Errorf("filtered image has no contrast (min == max == %d), cannot normalize", max)
	}

	out := &pgm.Image{
		Width:  image.Width,
		Height: image.Height,
		Pixels: make([][]uint8, image.Height),
	}
	for y := range out.Pixels {
		out.Pixels[y] = make([]uint8, image.Width)
	}

	scale := float64(pgm.MaxBrightness) / float64(max-min)
	for y := 1; y < image.Height-1; y++ {
		for x := 1; x < image.Width-1; x++ {
			out.Pixels[y][x] = uint8(scale * float64(raw[y][x]-min))
		}
	}
	return out, nil
}
