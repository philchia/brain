package brain

// Classifier used for classify
type Classifier interface {
	Fit(x [][]float64, y []string)
	Predict(x []float64) string
}

// Regressor used for regress
type Regressor interface {
	Fit(x [][]float64, y []float64)
	Predict(x []float64) float64
}
