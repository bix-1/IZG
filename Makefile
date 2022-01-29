# File: Makefile
# Brief: Makefile for IZG project at FIT BUT 2020
#
# Project: Software implementation of a basic GPU
#
# Authors: Jakub Bartko    xbartk07@stud.fit.vutbr.cz

unzip:
	unzip izgProject.zip -d src
	cp gpu.* src/student
	cd src/build && cmake ..
	cd src/build && make

run:
	./src/build/izgProject

clean:
	rm -rf src
