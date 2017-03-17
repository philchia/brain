package brain

// Classifier used for classify
type Classifier interface {
	Fit(x [][]float64, y []string)
	Predict(x []float64) string
}

// Regress used for regress
type Regress interface {
	Fit(x [][]float64, y []float64)
	Predict(x []float64) float64
}
