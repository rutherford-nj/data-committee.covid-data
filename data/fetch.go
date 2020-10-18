package main

import (
	"io/ioutil"
	"net/http"
)

func fetch(url string) []byte {
	resp, err := http.Get(url)
	if err != nil {
		panic(err)
	}

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}
	return body
}

func main() {
	d := fetch("https://api.covidtracking.com/v1/us/daily.json")
	writeCovidTracking(d, "csv/covid_tracking_us.csv")

	d = fetch("https://api.covidtracking.com/v1/states/nj/daily.json")
	writeCovidTracking(d, "csv/covid_tracking_nj.csv")

	d = fetch("https://docs.google.com/spreadsheets/d/e/2PACX-1vS00GBGJKB0Xwtru3Rn5WrPqur19j--CibdM5R1tbnis0W_Bp18EmLFkJJc5sG4dwvMyqCorSVhHwik/pub?output=csv")
	writeRutherfordData(d, "csv/rutherford_data.csv")

	d = fetch("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")
	writeCountyData(d, "csv/nytimes_nj_counties.csv")
}
