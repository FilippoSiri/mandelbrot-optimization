#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

#ifdef DOUBLE
typedef double __ftype;
#else
typedef float __ftype;
#endif

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#ifndef RESOLUTION
#define RESOLUTION 1000
#endif

#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((__ftype)RATIO_X / WIDTH)

#ifndef ITERATIONS
#define ITERATIONS 1000 // Maximum number of iterations
#endif

#ifndef THREAD_NO
#define THREAD_NO 24
#endif

#ifndef OMP_SCHEDULE
#define OMP_SCHEDULE dynamic
#endif

using namespace std;

void print_m128i(__m128i mm)
{
    int *mm_int = (int *)&mm;
    for (int i = 0; i < 4; i++)
    {
        printf("%08x ", mm_int[i]);
    }
    printf("\n");
}

void print_m256d(__m256d mm)
{
    double *mm_double = (double *)&mm;
    for (int i = 0; i < 4; i++)
    {
        printf("%lf ", mm_double[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    #ifdef DOUBLE
    long long int *const image = new long long int[HEIGHT * WIDTH];
    #else
    int* const image = new int[HEIGHT * WIDTH];
    #endif
    const auto start = chrono::steady_clock::now();

    const int block_count = THREAD_NO * 10;

    const int block_size = HEIGHT * WIDTH / block_count;

#ifdef DOUBLE

    __m256d step = _mm256_set1_pd(STEP);
    __m256d min_x = _mm256_set1_pd(MIN_X);
    __m256d min_y = _mm256_set1_pd(MIN_Y);

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(THREAD_NO)
    for (int pos = 0; pos < HEIGHT * WIDTH; pos += 4)
    {

        __m256d c_re = _mm256_set_pd(
            (pos + 3) % WIDTH,
            (pos + 2) % WIDTH,
            (pos + 1) % WIDTH,
            (pos + 0) % WIDTH);

#ifdef FMA
        c_re = _mm256_fmadd_pd(c_re, step, min_x);
#else
        c_re = _mm256_mul_pd(c_re, step);
        c_re = _mm256_add_pd(c_re, min_x);
#endif

        __m256d c_im = _mm256_set_pd(
            (pos + 3) / WIDTH,
            (pos + 2) / WIDTH,
            (pos + 1) / WIDTH,
            (pos + 0) / WIDTH);

#ifdef FMA
        c_im = _mm256_fmadd_pd(c_im, step, min_y);
#else
        c_im = _mm256_mul_pd(c_im, step);
        c_im = _mm256_add_pd(c_im, min_y);
#endif

        // set vectors to 0
        __m256d z_re = _mm256_setzero_pd();
        __m256d z_im = _mm256_setzero_pd();

        // We use `__m256i` and `long long`s instead of `__m128i` and regular `int`s for both `results` and `mask` to use registers
        // of the same width of the `double` vectors in order to perform an optimization in the update of
        // the result values inside the inner loop

        // Keep the result in a register  and use store to push it to RAM only once
        __m256i results = _mm256_set1_epi64x(0);

        // Initialize mask to all ones
        __m256i mask = _mm256_set1_epi64x(-1);

        // z = z^2 + c
        for (int i = 1; i <= ITERATIONS; i++)
        {
            // xy	=	(a+ib)(c+id)
            // 	    =	(ac-bd)+i(ad+bc).
            // a == c, b == d
            // ==> x * x = (a * a - b * b) + i (2 * a * b)

#ifdef FMA
            __m256d z2_re = _mm256_fmsub_pd(z_re, z_re, _mm256_mul_pd(z_im, z_im));
#else
            __m256d z2_re = _mm256_mul_pd(z_re, z_re);
            __m256d tmp = _mm256_mul_pd(z_im, z_im);
            z2_re = _mm256_sub_pd(z2_re, tmp);
#endif

            __m256d z2_im = _mm256_mul_pd(z_re, z_im);

// z = z^2 + c;
// => z2 + c
#ifdef FMA
            z_im = _mm256_fmadd_pd(_mm256_set1_pd(2.0), z2_im, c_im);
#else
            z2_im = _mm256_add_pd(z2_im, z2_im);
            z_im = _mm256_add_pd(z2_im, c_im);
#endif

            z_re = _mm256_add_pd(z2_re, c_re);

// |z|2 = x2 + y2.
#ifdef FMA
            __m256d abs2 = _mm256_fmadd_pd(z_re, z_re, _mm256_mul_pd(z_im, z_im));
#else
            __m256d abs2 = _mm256_add_pd(_mm256_mul_pd(z_re, z_re), _mm256_mul_pd(z_im, z_im));
#endif

            // image[pos] = should_update * i + (1 - should_update) * image[pos]
            __m256d abs2_gt_4 = _mm256_cmp_pd(abs2, _mm256_set1_pd(4.0), _CMP_GT_OQ);

            __m256i current_step = _mm256_set1_epi64x(i);

            // image[pos] = image[pos] || (abs2_gt_4[pos] && mask[pos] && current_step[pos])
            // mask[pos] = mask[pos] && (mask[pos] ^ abs2_gt_4[pos])
            results = _mm256_or_si256(results, _mm256_and_si256(_mm256_and_si256(abs2_gt_4, mask), current_step));
            mask = _mm256_and_si256(mask, _mm256_xor_si256(mask, abs2_gt_4));

            int all_diverge_mask = _mm256_movemask_pd(abs2_gt_4);

            // If all of the image pixels have diverged, then break out of the loop
            if (all_diverge_mask == 0xF)
            {
                break;
            }
        }

        _mm256_store_si256((__m256i *)&image[pos], results);
    }

#else

    __m256 step = _mm256_set1_ps(STEP);
    __m256 min_x = _mm256_set1_ps(MIN_X);
    __m256 min_y = _mm256_set1_ps(MIN_Y);

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(THREAD_NO)
    for (int pos = 0; pos < HEIGHT * WIDTH; pos += 8)
    {
        __m256 c_re = _mm256_set_ps(
            (pos + 7) % WIDTH,
            (pos + 6) % WIDTH,
            (pos + 5) % WIDTH,
            (pos + 4) % WIDTH,
            (pos + 3) % WIDTH,
            (pos + 2) % WIDTH,
            (pos + 1) % WIDTH,
            (pos + 0) % WIDTH);

#ifdef FMA
        c_re = _mm256_fmadd_ps(c_re, step, min_x);
#else
        c_re = _mm256_mul_ps(c_re, step);
        c_re = _mm256_add_ps(c_re, min_x);
#endif

        __m256 c_im = _mm256_set_ps(
            (pos + 7) / WIDTH,
            (pos + 6) / WIDTH,
            (pos + 5) / WIDTH,
            (pos + 4) / WIDTH,
            (pos + 3) / WIDTH,
            (pos + 2) / WIDTH,
            (pos + 1) / WIDTH,
            (pos + 0) / WIDTH);

#ifdef FMA
        c_im = _mm256_fmadd_ps(c_im, step, min_y);
#else
        c_im = _mm256_mul_ps(c_im, step);
        c_im = _mm256_add_ps(c_im, min_y);
#endif

        // set vectors to 0
        __m256 z_re = _mm256_setzero_ps();
        __m256 z_im = _mm256_setzero_ps();

        __m256i results = _mm256_setzero_si256();

        // Initialize mask to all ones
        __m256i mask = _mm256_set1_epi32(-1);

        // z = z^2 + c
        for (int i = 1; i <= ITERATIONS; i++)
        {
#ifdef FMA
            __m256 z2_re = _mm256_fmsub_ps(z_re, z_re, _mm256_mul_ps(z_im, z_im));
#else
            __m256 z2_re = _mm256_mul_ps(z_re, z_re);
            __m256 tmp = _mm256_mul_ps(z_im, z_im);
            z2_re = _mm256_sub_ps(z2_re, tmp);
#endif

            __m256 z2_im = _mm256_mul_ps(z_re, z_im);

// z = z^2 + c;
// => z2 + c
#ifdef FMA
            z_im = _mm256_fmadd_ps(_mm256_set1_ps(2.0), z2_im, c_im);
#else
            z2_im = _mm256_add_ps(z2_im, z2_im);
            z_im = _mm256_add_ps(z2_im, c_im);
#endif

            z_re = _mm256_add_ps(z2_re, c_re);

// |z|2 = x2 + y2.
#ifdef FMA
            __m256 abs2 = _mm256_fmadd_ps(z_re, z_re, _mm256_mul_ps(z_im, z_im));
#else
            __m256 abs2 = _mm256_add_ps(_mm256_mul_ps(z_re, z_re), _mm256_mul_ps(z_im, z_im));
#endif

            // image[pos] = should_update * i + (1 - should_update) * image[pos]
            __m256 abs2_gt_4 = _mm256_cmp_ps(abs2, _mm256_set1_ps(4.0), _CMP_GT_OQ);

            __m256i current_step = _mm256_set1_epi32(i);

            // image[pos] = image[pos] || (abs2_gt_4[pos] && mask[pos] && current_step[pos])
            // mask[pos] = mask[pos] && (mask[pos] ^ abs2_gt_4[pos])
            results = _mm256_or_si256(results, _mm256_and_si256(_mm256_and_si256(abs2_gt_4, mask), current_step));
            mask = _mm256_and_si256(mask, _mm256_xor_si256(mask, abs2_gt_4));

            int all_diverge_mask = _mm256_movemask_ps(abs2_gt_4);

            // If all of the image pixels have diverged, then break out of the loop
            if (all_diverge_mask == 0xFF)
            {
                break;
            }
        }

        _mm256_store_si256((__m256i *)&image[pos], results);
    }
#endif

    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms." << endl;

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image; // It's here for coding style, but useless
    return 0;
}
