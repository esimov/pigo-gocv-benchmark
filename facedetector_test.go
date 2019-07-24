package main

import (
	"image"
	"io/ioutil"
	"log"
	"testing"

	pigo "github.com/esimov/pigo/core"
	"gocv.io/x/gocv"
)

const (
	haarCascadeFile = "haarcascade_frontalface_default.xml"
	pigoCascadeFile = "facefinder.bin"
)

func BenchmarkGoCV(b *testing.B) {
	img := gocv.IMRead("sample.jpg", gocv.IMReadUnchanged)
	if img.Cols() == 0 || img.Rows() == 0 {
		b.Fatalf("Unable to read image into file")
	}

	classifier := gocv.NewCascadeClassifier()
	if !classifier.Load(haarCascadeFile) {
		b.Fatalf("Error reading cascade file: %v\n", haarCascadeFile)
	}

	var rects []image.Rectangle
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		rects = classifier.DetectMultiScale(img)
	}
	_ = rects
}

func BenchmarkPIGO(b *testing.B) {
	src, err := pigo.GetImage("sample.jpg")
	if err != nil {
		log.Fatalf("Error reading the source file: %s", err)
	}

	pixs := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	cParams := pigo.CascadeParams{
		MinSize:     20,
		MaxSize:     1000,
		ShiftFactor: 0.2,
		ScaleFactor: 1.1,
		ImageParams: pigo.ImageParams{
			Pixels: pixs,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	cf, err := ioutil.ReadFile(pigoCascadeFile)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %v", err)
	}

	pg := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := pg.Unpack(cf)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %s", err)
	}

	var dets []pigo.Detection
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		pixs := pigo.RgbToGrayscale(src)
		cParams.Pixels = pixs
		// Run the classifier over the obtained leaf nodes and return the detection results.
		// The result contains quadruplets representing the row, column, scale and detection score.
		dets = classifier.RunCascade(cParams, 0.0)
		// Calculate the intersection over union (IoU) of two clusters.
		dets = classifier.ClusterDetections(dets, 0.1)
	}
	_ = dets
}
