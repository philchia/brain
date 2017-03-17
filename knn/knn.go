package knn

import (
	"sort"

	"github.com/philchia/brain"
)

// KNeighborsClassifier returns a new Brain.
func KNeighborsClassifier(k int) brain.Classifier {
	brain := &knn{
		k: k,
	}
	return brain
}

type knn struct {
	k      int
	data   [][]float64
	target []string
}

type distance struct {
	dis    float64
	target string
}

func (c *knn) Fit(x [][]float64, y []string) {
	c.data = x
	c.target = y
}

func (c *knn) Predict(x []float64) string {
	var distances []distance
	for j := range c.data {
		dis := distance{
			target: c.target[j],
		}
		for i := range x {
			dis.dis += (c.data[j][i] - x[i]) * (c.data[j][i] - x[i])
		}
		distances = append(distances, dis)
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].dis < distances[j].dis
	})

	var majority = distances[0].target
	var count = 1
	for i := 1; i < c.k; i++ {
		if distances[i].target == majority {
			count++
		} else {
			if count == 0 {
				majority = distances[i].target
				count = 1
			} else {
				count--
			}
		}
	}
	return majority
}

// KNeighborsRegressor returns a new Brain.
func KNeighborsRegressor(k int) brain.Regressor {
	brain := &knnRegressor{
		k: k,
	}
	return brain
}

type knnRegressor struct {
	k      int
	data   [][]float64
	target []float64
}

type distanceRegressor struct {
	dis    float64
	target float64
}

func (r *knnRegressor) Fit(x [][]float64, y []float64) {
	r.data = x
	r.target = y
}

func (r *knnRegressor) Predict(x []float64) float64 {
	var distances []distanceRegressor
	for j := range r.data {
		dis := distanceRegressor{
			target: r.target[j],
		}
		for i := range x {
			dis.dis += (r.data[j][i] - x[i]) * (r.data[j][i] - x[i])
		}
		distances = append(distances, dis)
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].dis < distances[j].dis
	})

	var sum float64

	for i := 0; i < r.k; i++ {
		sum += distances[i].target
	}
	return sum / float64(r.k)
}
