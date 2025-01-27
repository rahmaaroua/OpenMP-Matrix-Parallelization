#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Partie 1 : Allocation et génération des matrices
float** allocate_matrix(int size) {
    float** matrix = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (float*)malloc(size * sizeof(float));
    }
    return matrix;
}

void free_matrix(float** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void generate_matrix(float** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (float)(rand() % 100);
        }
    }
}

// Produit matriciel séquentiel
void matrix_multiply_sequential(float** A, float** B, float** C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Wrapper function for sequential multiplication
void matrix_multiply_sequential_wrapper(float** A, float** B, float** C, int size, int schedule_type, int num_threads) {
    matrix_multiply_sequential(A, B, C, size);
}

// Partie 2 : Produit matriciel parallèle avec OpenMP
void matrix_multiply_parallelI(float** A, float** B, float** C, int size, int schedule_type, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Choisir la stratégie de parallélisation
    switch (schedule_type) {
        case 0:
            #pragma omp parallel for schedule(static)
            break;
        case 1:
            #pragma omp parallel for schedule(dynamic)
            break;
        case 2:
            #pragma omp parallel for schedule(guided)
            break;
        default:
            #pragma omp parallel for schedule(static)
            break;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Mesure du temps d'exécution
double measure_time(void (*matrix_function)(float**, float**, float**, int, int, int), float** A, float** B, float** C, int size, int schedule_type, int num_threads) {
    double start_time = omp_get_wtime();
    matrix_function(A, B, C, size, schedule_type, num_threads);
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// Fonction principale
int main() {
    int sizes[] = {512, 1024, 2048}; // Tailles des matrices
    int num_threads[] = {1, 2, 4, 8}; // Nombre de threads
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_thread_configs = sizeof(num_threads) / sizeof(num_threads[0]);

    srand(time(NULL));

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];

        // Allocation des matrices
        float** A = allocate_matrix(size);
        float** B = allocate_matrix(size);
        float** C = allocate_matrix(size);

        // Initialisation des matrices
        generate_matrix(A, size);
        generate_matrix(B, size);

        printf("\nTaille de la matrice: %dx%d\n", size, size);

        // Exécution et mesure du produit matriciel séquentiel
        double seq_time = measure_time(matrix_multiply_sequential_wrapper, A, B, C, size, 0, 1);
        printf("Temps séquentiel: %f secondes\n", seq_time);

        // Exécution et mesure des configurations parallèles
        for (int t = 0; t < num_thread_configs; t++) {
            int threads = num_threads[t];
            printf("\nNombre de threads: %d\n", threads);

            for (int schedule_type = 0; schedule_type < 3; schedule_type++) {
                char* schedule_name = (schedule_type == 0) ? "Statique" : (schedule_type == 1) ? "Dynamique" : "Par blocs";
                double par_timeI = measure_time(matrix_multiply_parallelI, A, B, C, size, schedule_type, threads);
                printf("Stratégie: %s, Temps: %f secondes, Speedup: %f\n", schedule_name, par_timeI, seq_time / par_timeI);
            }
        }

        // Libération des matrices
        free_matrix(A, size);
        free_matrix(B, size);
        free_matrix(C, size);
    }

    return 0;
}
