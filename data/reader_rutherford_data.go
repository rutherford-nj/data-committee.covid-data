package main

import (
	"os"
)

func writeRutherfordData(csvData []byte, path string) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	f.Write(csvData)
}
