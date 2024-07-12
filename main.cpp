#include "EcuacionCalor.hpp"
#include <iostream>
#include <vector>
#include <sys/time.h>

double seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;

  return sec;
}

int main() {
    // Parámetros de simulación
    double a = 10.0;         // Límite superior del dominio en x
    double b = 10.0;         // Límite superior del dominio en y
    int g = 100;             // Número de grillas
    int iteraciones = 10;    // Número de iteraciones
    double k = 1.0;          // Parámetro k
    double t = 1.0;          // Tiempo
    std::string documento = "resultado.csv"; // Nombre del archivo CSV de salida

    
    double time_1 = seconds();
    
    auto func = [](double x, double y) { return x * y; }; // Función de ejemplo, ajusta según sea necesario

    // Precalcular los coeficientes B0n y Bmn
    auto [B0n_n, Bmn_mn] = precalc_inte(a, b, g, iteraciones, func);

    // Calcular la onda de calor en 2D
    std::vector<double> x(g), y(g);
    std::vector<std::vector<double>> u(g, std::vector<double>(g, 0.0));
    onda2D(a, b, g, iteraciones, k, t, B0n_n, Bmn_mn);

    // Actualizar las coordenadas x e y
    for (int i = 0; i < g; ++i) {
        x[i] = i * a / (g - 1);
        y[i] = i * b / (g - 1);
    }

    // Exportar los resultados a un archivo CSV
    exportarCSV(x, y, u, documento);

    std::cout << "Simulación completada. Resultados guardados en " << documento << std::endl;
    
    double time_2 = seconds();
    std::cout << "# Time: " << time_2 - time_1 << std::endl;
    return 0;
}

