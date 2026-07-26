package pgm

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

const MaxBrightness = 255

type Image struct {
	Width  int
	Height int
	Pixels [][]uint8
}

// load and reads PGM (P5)
func Load(fileName string) (*Image, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, fmt.Errorf("opening %q: %w", fileName, err)
	}
	defer f.Close()

	r := bufio.NewReader(f)

	magic, err := readToken(r)
	if err != nil {
		return nil, fmt.Errorf("reading magic number: %w", err)
	}
	if magic != "P5" {
		return nil, fmt.Errorf("unsupported format %q, expected P5", magic)
	}

	width, err := readIntToken(r)
	if err != nil {
		return nil, fmt.Errorf("reading width: %w", err)
	}
	height, err := readIntToken(r)
	if err != nil {
		return nil, fmt.Errorf("reading height: %w", err)
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimension %dx%d", width, height)
	}

	maxGray, err := readIntToken(r)
	if err != nil {
		return nil, fmt.Errorf("reading max gray: %w", err)
	}
	if maxGray != MaxBrightness {
		return nil, fmt.Errorf("unsupported max gray %d, expected %d", maxGray, MaxBrightness)
	}

	pixels := make([][]uint8, height)
	for y := 0; y < height; y++ {
		pixels[y] = make([]uint8, width)
		if _, err := io.ReadFull(r, pixels[y]); err != nil {
			return nil, fmt.Errorf("reading pixel row %d:%w", y, err)
		}
	}
	return &Image{Width: width, Height: height, Pixels: pixels}, nil
}

func readToken(r *bufio.Reader) (string, error) {
	var sb strings.Builder

	// skip whitespace and any comment lines
	for {
		b, err := r.ReadByte()
		if err != nil {
			return "", err
		}
		if b == '#' {
			if _, err := r.ReadString('\n'); err != nil {
				return "", err
			}
			continue
		}
		if isSpace(b) {
			continue
		}
		sb.WriteByte(b)
		break
	}

	// read the token
	for {
		b, err := r.ReadByte()
		if err != nil {
			break
		}
		if b == '#' {
			r.ReadString('\n')
			break
		}
		if isSpace(b) {
			break
		}
		sb.WriteByte(b)
	}
	return sb.String(), nil
}

func Save(filename string, img *Image) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("creating %q: %w", filename, err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)

	header := fmt.Sprintf("P5\n# Created by Image Processing\n%d %d\n%d\n",
		img.Width, img.Height, MaxBrightness)
	if _, err := w.WriteString(header); err != nil {
		return fmt.Errorf("writing header: %w", err)
	}

	for y := 0; y < img.Height; y++ {
		if _, err := w.Write(img.Pixels[y]); err != nil {
			return fmt.Errorf("writing pixel row %d: %w", y, err)
		}
	}

	if err := w.Flush(); err != nil {
		return fmt.Errorf("flushing output: %w", err)
	}

	return nil
}

func readIntToken(r *bufio.Reader) (int, error) {
	tok, err := readToken(r)
	if err != nil {
		return 0, err
	}
	n, err := strconv.Atoi(tok)
	if err != nil {
		return 0, fmt.Errorf("parsing %q as integer: %w", tok, err)
	}
	return n, nil
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}
