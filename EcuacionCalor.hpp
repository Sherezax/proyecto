#ifndef EcuacionCalor_HPP
#define EcuacionCalor_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>  // Necesario para std::function
#include <fstream>

// Declaración de la función f(x, y)
double f(double x, double y, std::function<double(double, double)> func);

// Declaración de la función para calcular B0n
double calcB0n(double a, double b, int g, int n, std::function<double(double, double)> f);

// Declaración de la función para calcular Bmn
double calcBmn(double a, double b, int g, int m, int n, std::function<double(double, double)> f);

// Declaración de la función para precalcular los coeficientes B0n y Bmn
std::pair<std::vector<double>, std::vector<std::vector<double>>> precalc_inte(double a, double b, int g, int iteraciones, std::function<double(double, double)> f);

// Declaración de la función para calcular la onda de calor en 2D
void onda2D(double a, double b, int g, int iteraciones, double k, double t, const std::vector<double>& B0n_n, const std::vector<std::vector<double>>& Bmn_mn);

// Declaración de la función para exportar los datos a un archivo CSV
void exportarCSV(const std::vector<double>& x, const std::vector<double>& y, const std::vector<std::vector<double>>& u, const std::string& filename);

#endif
