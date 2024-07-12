# Archivo: animacion.gp

# Establecer el terminal y el tamaño de salida (puedes ajustar el tamaño según tus necesidades)
set terminal png size 800,600

# Establecer la salida de la primera imagen
set output 'frame_001.png'

# Configuración del gráfico
set pm3d map
set palette rgbformula 33,13,10  # Configura la paleta de colores

set xlabel 'X'
set ylabel 'Y'
set title 'Mapa de calor 2D'

# Ajustar el rango de los ejes x e y según tus datos
set xrange [0:10]
set yrange [0:10]

# Definir el parámetro de tiempo inicial (o cualquier otra dimensión que estés simulando)
t = 0.0

# Plot inicial
plot 'resultado.csv' using 1:2:(column(3)) with image notitle
# Archivo: animacion.gp (continuación)

# Bucle para generar múltiples imágenes
do for [i=1:100] {
    t = i * 0.1  # Ajusta t según el avance de la simulación

    # Establecer el nombre de salida para la imagen actual
    fname = sprintf('frame_%03d.png', i)
    set output fname

    # Plot con los datos correspondientes a la iteración actual
    plot 'resultado.csv' using 1:2:(column(3)) with image notitle
}

