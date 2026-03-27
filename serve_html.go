package main

import (
	"log"
	"net/http"
)

func main() {
	directory := "./html"
	port := ":8080"

	fileServer := http.FileServer(http.Dir(directory))

	log.Printf("Serving %s on HTTP port %s\n", directory, port)

	if err := http.ListenAndServe(port, fileServer); err != nil {
		log.Fatal(err)
	}
}
