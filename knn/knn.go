package knn

import (
	"sort"

	"github.com/philchia/brain"
)

// New returns a new Brain.
func New(k int) brain.Classifier {
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
