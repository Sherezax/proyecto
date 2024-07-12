#include <iostream>
#include "EcuacionCalor.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <fstream>
#include <omp.h>

// Implementación de la función f(x, y)
double f(double x, double y, std::function<double(double, double)> func) {
    return func(x, y);
}

// Implementación de la función para calcular B0n
double calcB0n(double a, double b, int g, int n, std::function<double(double, double)> f) {
    double integral = 0.0;
    double dx = a / (g - 1); // Corregido para g - 1
    double dy = b / (g - 1); // Corregido para g - 1
    #pragma omp parallel for reduction(+:integral)
    for (int i = 0; i < g; ++i) {
        double x = dx * i;
        for (int j = 0; j < g; ++j) {
            double y = dy * j;
            integral += f(x, y) * sin(M_PI * n * y / b) * dx * dy;
        }
    }
    return 4 * integral / (a * b);
}

// Implementación de la función para calcular Bmn
double calcBmn(double a, double b, int g, int m, int n, std::function<double(double, double)> f) {
    double integral = 0.0;
    double dx = a / (g - 1); // Corregido para g - 1
    double dy = b / (g - 1); // Corregido para g - 1
    #pragma omp parallel for reduction(+:integral)
    for (int i = 0; i < g; ++i) {
        double x = dx * i;
        for (int j = 0; j < g; ++j) {
            double y = dy * j;
            integral += f(x, y) * sin(M_PI * n * y / b) * cos(M_PI * m * x / a) * dx * dy;
        }
    }
    return 4 * integral / (a * b);
}

// Implementación de la función para precalcular los coeficientes B0n y Bmn
std::pair<std::vector<double>, std::vector<std::vector<double>>> precalc_inte(double a, double b, int g, int iteraciones, std::function<double(double, double)> f) {
    std::vector<double> B0n_n(iteraciones);
    std::vector<std::vector<double>> Bmn_mn(iteraciones, std::vector<double>(iteraciones));
    
    #pragma omp parallel for
    for (int n = 1; n <= iteraciones; ++n) {
        B0n_n[n - 1] = calcB0n(a, b, g, n, f);
    }
    #pragma omp parallel for collapse(2)
    for (int m = 1; m <= iteraciones; ++m) {
        for (int n = 1; n <= iteraciones; ++n) {
            Bmn_mn[m - 1][n - 1] = calcBmn(a, b, g, m, n, f);
        }
    }
    return {B0n_n, Bmn_mn};
}

// Implementación de la función para calcular la onda de calor en 2D
void onda2D(double a, double b, int g, int iteraciones, double k, double t, const std::vector<double>& B0n_n, const std::vector<std::vector<double>>& Bmn_mn) {
    std::vector<double> x(g), y(g);
    std::vector<std::vector<double>> u(g, std::vector<double>(g, 0.0));
    for (int i = 0; i < g; ++i) {
        x[i] = i * a / (g - 1); // Corregido para g - 1
        y[i] = i * b / (g - 1); // Corregido para g - 1
    }
    for (int n = 1; n <= iteraciones; ++n) {
        double B0n = B0n_n[n - 1];
        for (int i = 0; i < g; ++i) {
	u[i][0] += 0.5 * B0n * exp(-M_PI * M_PI * n * n * k * t / ( b * b)) * sin(M_PI * n * y[i] / b);
	}
    }
    for (int m = 1; m <= iteraciones; ++m) {
        for (int n = 1; n <= iteraciones; ++n) {
            double Bmn = Bmn_mn[m - 1][n - 1];
            for (int i = 0; i < g; ++i) {
                for (int j = 0; j < g; ++j) {
	        u[i][j] += Bmn * exp(-M_PI * M_PI * (m * m / ( a * a ) + n * n / ( b * b)) * k * t) * sin(M_PI * n * y[j] / b) * cos(M_PI * m * x[i] / a);
  		}
            }
        }
    }
}

void exportarCSV(const std::vector<double>& x, const std::vector<double>& y, const std::vector<std::vector<double>>& u, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << " para escritura." << std::endl;
        return;
    }

    // Escribir encabezados 
    file << "X,Y,";
    for (size_t i = 0; i < x.size(); ++i) {
        file << "Y" << i+1;
        if (i < x.size() - 1)
            file << ",";
    }
    file << "\n";

    // Escribir datos
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            file << x[i] << "," << y[j] << ",";
            file << u[j][i]; // El orden y[j][i] es debido a cómo se manejan las filas y columnas
            if (j < y.size() - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}

