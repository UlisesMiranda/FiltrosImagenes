#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace cv;
using namespace std;

vector<vector<float>> mascaraGaussiana(int mascSize, float sigma) {

    int limite = (mascSize - 1) / 2;
    float r, s, z, gaussResultado;
    float sum = 0.0;

    vector<vector<float>> mascara(mascSize, vector<float>(mascSize, 0));

    s = 2.0 * sigma * sigma;

    cout << "LOS VALORES DEL KERNEL DE GAUSS: " << endl;

    for (int x = -limite; x <= limite; x++) {
        for (int y = -limite; y <= limite; y++) {

            //FORMULA DE INTERNET
            r = sqrt(x * x + y * y);
            z = (exp(-(r * r) / s)) / (M_PI * s);
            gaussResultado = (exp(-(r * r) / s)) / (M_PI * s);
            mascara[x + limite][y + limite] = gaussResultado;

            cout << gaussResultado << endl;
            sum += gaussResultado;
        }
    }

    for (int i = 0; i < mascSize; i++) {
        for (int j = 0; j < mascSize; j++) {
            mascara[i][j] /= sum;
        }
    }

    return mascara;
}

int getMin(int array[], int N) {

    int min = 0;
    for (size_t i = 0; i < N; i++)
    {
        if (array[i] < min) {
            min = array[i];
        }
    }

    return min;
}
 

vector<vector<float>> mascaraGy() {

    vector<vector<float>> mascara(3, vector<float>(3, 0));

    mascara[0][0] = -1;
    mascara[0][1] = -2;
    mascara[0][2] = -1;

    mascara[1][0] = 0;
    mascara[1][1] = 0;
    mascara[1][2] = 0;

    mascara[2][0] = 1;
    mascara[2][1] = 2;
    mascara[2][2] = 1;

    return mascara;
}

vector<vector<float>> mascaraGx() {

    vector<vector<float>> mascara(3, vector<float>(3, 0));

    mascara[0][0] = -1;
    mascara[0][1] = 0;
    mascara[0][2] = 1;

    mascara[1][0] = -2;
    mascara[1][1] = 0;
    mascara[1][2] = 2;

    mascara[2][0] = -1;
    mascara[2][1] = 0;
    mascara[2][2] = 1;

    return mascara;
}

Mat matrizRelleno(int filas, int columnas, int mascSize)
{
    int diferenciaBordes = mascSize - 1;
    Mat matriz(filas + diferenciaBordes, columnas + diferenciaBordes, CV_8UC1);

    for (int i = 0; i < filas + diferenciaBordes; i++)
    {
        for (int j = 0; j < columnas + diferenciaBordes; j++)
        {
            matriz.at<uchar>(Point(j, i)) = uchar(0);
        }
    }

    return matriz;
}

Mat copiarImgARelleno(Mat bordes, Mat original, int mascSize)
{
    int diferenciaBordes = ((mascSize - 1) / 2);
    int filas = bordes.rows;
    int columnas = bordes.cols;

    for (int i = diferenciaBordes; i < filas - diferenciaBordes; i++)
    {
        for (int j = diferenciaBordes; j < columnas - diferenciaBordes; j++)
        {
            bordes.at<uchar>(Point(j, i)) = original.at<uchar>(Point(j - diferenciaBordes, i - diferenciaBordes));
        }
    }

    return bordes;
}

float convolucionPixel(Mat matrizConBordes, vector<vector<float>> mascara, int mascSize, int x, int y) {

    int limites = (mascSize - 1) / 2;

    float sumatoriaFiltro = 0.0;

    for (int i = -limites; i <= limites; i++) {
        for (int j = -limites; j <= limites; j++) {

            float valMascara = mascara[i + limites][j + limites];
            int coordY = y + j + limites;
            int coordX = x + i + limites;

            float valImagen = matrizConBordes.at<uchar>(coordY, coordX);

            sumatoriaFiltro += valMascara * valImagen;
        }
    }

    return sumatoriaFiltro;
}

Mat aplicarFiltroImagen(Mat imagenOriginal, Mat matrizConBordes, vector<vector<float>> mascara, int mascSize) {
    int filas = imagenOriginal.rows;
    int columnas = imagenOriginal.cols;

    Mat imagenFiltroAplicado(filas, columnas, CV_8UC1);

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            float val = abs(static_cast<int>(convolucionPixel(matrizConBordes, mascara, mascSize, i, j)));
            imagenFiltroAplicado.at<uchar>(Point(i, j)) = val;
        }
    }

    return imagenFiltroAplicado;
}

Mat imagenFiltroSobel(Mat imagenGx, Mat imagenGy) {

    int filas = imagenGx.rows;
    int columnas = imagenGy.cols;
    int umbral = 100;
    double intensidad;
    double valGx, valGy;

    Mat sobel(filas, columnas, CV_8UC1);

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {

            valGx = imagenGx.at<uchar>(Point(j, i));
            valGy = imagenGy.at<uchar>(Point(j, i));

            intensidad = sqrt(pow(valGx, 2) + pow(valGy, 2));

            sobel.at<uchar>(Point(j, i)) = uchar(intensidad);
        }
    }

    return sobel;
}

vector<vector<double>> calcularDirecciones(Mat imagenGx, Mat imagenGy) {

    double valX, valY;
    int filas = imagenGx.rows;
    int columnas = imagenGy.cols;
    double valGx, valGy;

    vector<vector<double>> direcciones(filas, vector<double>(columnas, 0));

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            valGx = imagenGx.at<uchar>(Point(j, i));
            valGy = imagenGy.at<uchar>(Point(j, i));

            direcciones[i][j] = (atan(valGy / valGx) * 180.0) / M_PI; //grados sexagesimales

            if (direcciones[i][j] < 0) { 
                direcciones[i][j] += 180;
            }
        }
    }

    return direcciones;
}

Mat nonMaxSupression(Mat imagenSobel, vector<vector<double>> direcciones) {
    int filas = imagenSobel.rows;
    int columnas = imagenSobel.cols;

    Mat imgNonMaxSupr(filas, columnas, CV_8UC1);

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            int primerLado = 255;
            int segundoLado = 255;

            //si el angulo es 0� o bien 180�, obtiene las intensidades de izquierda y derecha
            if ((0 <= direcciones[i][j] < 22.5) || (157.5 <= direcciones[i][j] <= 180)) {
                primerLado = imagenSobel.at<uchar>(Point(i, j + 1));
                segundoLado = imagenSobel.at<uchar>(Point(i, j - 1));
            }
                
            //para 45� las esquinas
            else if( 22.5 <= direcciones[i][j] < 67.5) {
                primerLado = imagenSobel.at<uchar>(Point(i + 1, j - 1));
                segundoLado = imagenSobel.at<uchar>(Point(i - 1, j + 1));
            }

            //para 90� arriba y abajo
            else if (67.5 <= direcciones[i][j] < 112.5) {
                primerLado = imagenSobel.at<uchar>(Point(i + 1, j));
                segundoLado = imagenSobel.at<uchar>(Point(i - 1, j));
            }

            //para 135� las otras esquinas 
            else if (112.5 <= direcciones[i][j] < 157.5) {
                primerLado = imagenSobel.at<uchar>(Point(i - 1, j - 1));
                segundoLado = imagenSobel.at<uchar>(Point(i + 1, j + 1));
            }

            //Si las intesidades de los lados son menores, se mantiene la intensidad, otro caso se vuelve cero
            if ((imagenSobel.at<uchar>(Point(i, j)) >= primerLado) && imagenSobel.at<uchar>(Point(i, j)) >= segundoLado) {
                imgNonMaxSupr.at<uchar>(Point(i, j)) = imagenSobel.at<uchar>(Point(i, j));
            }
            else {
                imgNonMaxSupr.at<uchar>(Point(i, j)) = uchar(0);
            }
        }
    }

    return imgNonMaxSupr;
}

int intensidadMaxima(Mat imagen) {
    int filas = imagen.rows;
    int columnas = imagen.cols;

    int max = 0;

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            if (imagen.at<uchar>(Point(i, j)) > max) {
                max = imagen.at<uchar>(Point(i, j));
            }
        }
    }

    return max;
}

Mat hysteresis(Mat imgNonMaxSupr, float upperThresholdPorcentaje, float lowThresholdPorcentaje) {
    int filas = imgNonMaxSupr.rows;
    int columnas = imgNonMaxSupr.cols;

    Mat imgHysteresis(filas, columnas, CV_8UC1);

    float upperThreshold, lowThreshold; 

    upperThreshold = intensidadMaxima(imgNonMaxSupr) * upperThresholdPorcentaje; //90% of max
    lowThreshold = upperThreshold * lowThresholdPorcentaje; //35% of upper

    //de acuerdo con las diapositivas
    int weak = lowThreshold;
    int strong = 255;
    int irrelevant = 0;


    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {

            //condiciones de ejemplo diapostivas
            if (imgNonMaxSupr.at<uchar>(Point(i, j)) >= upperThreshold) {
                imgHysteresis.at<uchar>(Point(i, j)) = strong;
            }
            /*else if (lowThreshold < imgNonMaxSupr.at<uchar>(Point(i, j)) < upperThreshold) {
                imgHysteresis.at<uchar>(Point(i, j)) = weak;
            }*/
            else{
                imgHysteresis.at<uchar>(Point(i, j)) = irrelevant;
            }
        }
    }

    return imgHysteresis;
}



int main()
{
    int mascSize = 0, filasImagen = 0, columnasImagen = 0;
    float sigma = 0.0;

    Mat imagenOriginal, imagenEscGrises;
    char imageName[] = "lena.png";

    cout << "Inserte el tama�o de su mascara cuadrada: ";
    cin >> mascSize;
    cout << "Inserte el valor de sigma: ";
    cin >> sigma;

    imagenOriginal = imread(imageName);
    // Error reading imagenOriginal validation
    if (!imagenOriginal.data)
    {
        cout << "Error al cargar la imagen: " << imageName << endl;
        exit(1);
    }

    filasImagen = imagenOriginal.rows;
    columnasImagen = imagenOriginal.cols;

    //Obtenemos las mascaras
    vector<vector<float>> mascGaussiana = mascaraGaussiana(mascSize, sigma);
    vector<vector<float>> Gx = mascaraGx();
    vector<vector<float>> Gy = mascaraGy();

    //Pasamos a escala de grises
    cvtColor(imagenOriginal, imagenEscGrises, COLOR_BGR2GRAY);

    //Realizamos la expansion de bordes de la imagen original
    Mat matrizConBordes = matrizRelleno(filasImagen, columnasImagen, mascSize);
    matrizConBordes = copiarImgARelleno(matrizConBordes, imagenEscGrises, mascSize);

    //Aplicamos el filtro gaussiano
    Mat imagenFiltroGaussiano = aplicarFiltroImagen(imagenEscGrises, matrizConBordes, mascGaussiana, mascSize);

    //Ecualizamos la imagen
    //Mat imagenFiltroGaussianoEcualizada = ecualizarImagen(imagenFiltroGaussiano);
    Mat imagenFiltroGaussianoEcualizada;
    equalizeHist(imagenFiltroGaussiano, imagenFiltroGaussianoEcualizada); 

    //Nuevamente creamos la expansion pero ahora con la imagen del filtro gaussiano
    matrizConBordes = matrizRelleno(filasImagen, columnasImagen, mascSize);
    matrizConBordes = copiarImgARelleno(matrizConBordes, imagenFiltroGaussianoEcualizada, mascSize);

    //Aplicamos las mascaras de sobel 
    Mat imagenFiltroAplicadoGx = aplicarFiltroImagen(imagenFiltroGaussianoEcualizada, matrizConBordes, Gx, 3);
    Mat imagenFiltroAplicadoGy = aplicarFiltroImagen(imagenFiltroGaussianoEcualizada, matrizConBordes, Gy, 3);

    //Aplicamos el modulo de G de la sumatoria de ambas imagenes Gx y Gy
    Mat imgSobel = imagenFiltroSobel(imagenFiltroAplicadoGy, imagenFiltroAplicadoGx);

    //Calculamos la matriz de direcciones
    vector<vector<double>> direcciones = calcularDirecciones(imagenFiltroAplicadoGx, imagenFiltroAplicadoGy);

    //Obtenemos el nonMaxSupression
    Mat imgNonMaxSupr = nonMaxSupression(imgSobel, direcciones);

    //Realizamos la hysteresis
    Mat imgHysteresis = hysteresis(imgNonMaxSupr, 0.9, 0.35);


    //Imprimimos
    namedWindow("Imagen original", WINDOW_AUTOSIZE);
    imshow("Imagen original", imagenOriginal);

    namedWindow("Imagen escala de grises", WINDOW_AUTOSIZE);
    imshow("Imagen escala de grises", imagenEscGrises);

    /*namedWindow("Imagen Bordes Extra", WINDOW_AUTOSIZE);
    imshow("Imagen Bordes Extra", matrizConBordes);*/

    namedWindow("Imagen filtro gaussiano", WINDOW_AUTOSIZE);
    imshow("Imagen filtro gaussiano", imagenFiltroGaussiano);

    namedWindow("Imagen filtro gaussiano ecualizado", WINDOW_AUTOSIZE);
    imshow("Imagen filtro gaussiano ecualizado", imagenFiltroGaussianoEcualizada);

    /*namedWindow("Imagen con filtro Gx", WINDOW_AUTOSIZE);
    imshow("Imagen con filtro Gx", imagenFiltroAplicadoGx);

    namedWindow("Imagen con filtro Gy", WINDOW_AUTOSIZE);
    imshow("Imagen con filtro Gy", imagenFiltroAplicadoGy);*/

    namedWindow("Imagen filtro sobel G", WINDOW_AUTOSIZE);
    imshow("Imagen filtro sobel G", imgSobel);

    namedWindow("Imagen nonMax", WINDOW_AUTOSIZE);
    imshow("Imagen nonMax", imgNonMaxSupr);

    namedWindow("Imagen hysteresis", WINDOW_AUTOSIZE);
    imshow("Imagen hysteresis", imgHysteresis);

    waitKey(0);

    return 1;
}