/*
 * LTE Max-Log-MAP turbo decoder - SSE recursions
 *
 * Copyright (C) 2015 Ettus Research LLC
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Tom Tsou <tom.tsou@ettus.com>
 */

 /*
  * LTE parity and systematic output bits
  *
  * Bits are ordered consecutively from state 0 with even-odd referring to 0 and
  * 1 transitions respectively. Only bits for the upper paths (transitions from
  * states 0-3) are shown. Trellis symmetry and anti-symmetry applies, so for
  * lower paths (transitions from states 4-7), systematic bits will be repeated
  * and parity bits will repeated with inversion.
  */
#define LTE_SYSTEM_OUTPUT	1, -1, -1, 1, -1, 1, 1, -1
#define LTE_PARITY_OUTPUT	-1, 1, 1, -1, -1, 1, 1, -1

  /*
   * Shuffled systematic bits
   *
   * Rearranged systematic output based on the ordering of branch metrics used for
   * forward metric calculation. This avoids shuffling within the iterating
   * portion. The shuffle pattern is shown below. After shuffling, only upper bits
   * are used - lower portion is repeated and inverted.
   *
   * 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
   */
#define LTE_SYSTEM_FW_SHUFFLE	-1, 1, -1, 1, 1, -1, 1, -1
#define LTE_PARITY_FW_SHUFFLE	-1, -1, 1, 1, 1, 1, -1, -1
#define LTE_PARITY_BW_SHUFFLE	LTE_PARITY_FW_SHUFFLE

   /*
	* Max-Log-MAP Forward Recursion
	*
	* Forward trellis traversal of Max-Log-MAP variation of BCJR algorithm. This
	* includes generating branch metrics (gamma) and foward metrics (alpha). Branch
	* metrics are stored in shuffled form (i.e. ordering that matches the forward
	* metric output).
	*/
#define FW_SHUFFLE_MASK0 13, 12, 9, 8, 5, 4, 1, 0, 13, 12, 9, 8, 5, 4, 1, 0
#define FW_SHUFFLE_MASK1 15, 14, 11, 10, 7, 6, 3, 2, 15, 14, 11, 10, 7, 6, 3, 2

	/*
	 * Max-Log-MAP Backard Recursion
	 *
	 * Reverse trellis traversal of Max-Log-MAP variation of BCJR algorithm. This
	 * includes generating reverse metrics (beta) and outputing log likihood ratios.
	 * There is only a single buffer for backward metrics; the previous metric is
	 * overwritten with the new metric.
	 *
	 * Note: Because the summing pattern of 0 and 1 LLR bits is coded into the
	 * interleaver, this implementation is specific to LTE generator polynomials.
	 * The shuffle mask will need to be recomputed for non-LTE generators.
	 */
#define LV_BW_SHUFFLE_MASK0 7, 6, 15, 14, 13, 12, 5, 4, 3, 2, 11, 10, 9, 8, 1, 0
#define LV_BW_SHUFFLE_MASK1 15, 14, 7, 6, 5, 4, 13, 12, 11, 10, 3, 2, 1, 0, 9, 8

#ifdef HAVE_SSE3
#include <stdint.h>
#include <math.h>
#if !defined(__MACH__)
#include <malloc.h>
#endif
#include <emmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>

#if defined(HAVE_SSE4_1) || defined(HAVE_SSE41)
#ifdef HAVE_AVX2
#include <immintrin.h>
#endif
#include <smmintrin.h>
#define MAXPOS(M0,M1,M2) \
{ \
	M1 = _mm_set1_epi16(32767); \
	M2 = _mm_sub_epi16(M1, M0); \
	M2 = _mm_minpos_epu16(M2); \
	M2 = _mm_sub_epi16(M1, M2); \
}
#else
#define MAXPOS(M0,M1,M2) \
{ \
	M1 = _mm_shuffle_epi32(M0, _MM_SHUFFLE(0, 0, 3, 2)); \
	M2 = _mm_max_epi16(M0, M1); \
	M1 = _mm_shufflelo_epi16(M2, _MM_SHUFFLE(0, 0, 3, 2)); \
	M2 = _mm_max_epi16(M2, M1); \
	M1 = _mm_shufflelo_epi16(M2, _MM_SHUFFLE(0, 0, 0, 1)); \
	M2 = _mm_max_epi16(M2, M1); \
}
#endif



static inline int16_t gen_fw_metrics(int16_t* bm, int8_t x, int8_t z,
	int16_t* sums_p, int16_t* sums_c, int16_t le)
{
	__m128i m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13;

	m0 = _mm_set1_epi16(x);
	m1 = _mm_set1_epi16(z);
	m2 = _mm_set1_epi16(le);
	m3 = _mm_setzero_si128();
	m4 = _mm_set_epi16(LTE_SYSTEM_FW_SHUFFLE);
	m5 = _mm_set_epi16(LTE_PARITY_FW_SHUFFLE);

	/* Branch metrics */
	m6 = _mm_sign_epi16(m0, m4);
	m7 = _mm_sign_epi16(m1, m5);
	m8 = _mm_sign_epi16(m2, m4);
	m8 = _mm_srai_epi16(m8, 1);

	m6 = _mm_adds_epi16(m6, m7);
	m6 = _mm_adds_epi16(m6, m8);
	m7 = _mm_subs_epi16(m3, m6);

	/* Pre-interleave for backward recursion */
	m8 = _mm_unpacklo_epi16(m6, m7);
	_mm_store_si128((__m128i*) bm, m8);

	/* Forward metrics */
	m9 = _mm_load_si128((__m128i*) sums_p);
	m10 = _mm_set_epi8(FW_SHUFFLE_MASK0);
	m11 = _mm_set_epi8(FW_SHUFFLE_MASK1);

	m12 = _mm_shuffle_epi8(m9, m10);
	m13 = _mm_shuffle_epi8(m9, m11);
	m12 = _mm_adds_epi16(m12, m6);
	m13 = _mm_adds_epi16(m13, m7);

	m0 = _mm_max_epi16(m12, m13);
#ifdef HAVE_AVX2
	m1 = _mm_broadcastw_epi16(m0);
#else
	m1 = _mm_unpacklo_epi16(m0, m0);
	m1 = _mm_unpacklo_epi32(m1, m1);
	m1 = _mm_unpacklo_epi64(m1, m1);
#endif
	m0 = _mm_subs_epi16(m0, m1);

	_mm_store_si128((__m128i*) sums_c, m0);

	/* Return 32-bit integer with garbage in upper 16-bits */
	return _mm_cvtsi128_si32(m1);
}



static inline int16_t gen_bw_metrics(int16_t* bm, const int8_t z,
	int16_t* fw, int16_t* bw, int16_t norm)
{
	__m128i m0, m1, m3, m4, m5, m6, m9, m10, m11, m12, m13, m14, m15;

	m0 = _mm_set1_epi16(z);
	m1 = _mm_set_epi16(LTE_PARITY_BW_SHUFFLE);

	/* Partial branch metrics */
	m13 = _mm_sign_epi16(m0, m1);

	/* Backward metrics */
	m0 = _mm_load_si128((__m128i*) bw);
	m1 = _mm_load_si128((__m128i*) bm);
	m6 = _mm_load_si128((__m128i*) fw);
	m3 = _mm_set1_epi16(norm);

	m4 = _mm_unpacklo_epi16(m0, m0);
	m5 = _mm_unpackhi_epi16(m0, m0);
	m4 = _mm_adds_epi16(m4, m1);
	m5 = _mm_subs_epi16(m5, m1);

	m1 = _mm_max_epi16(m4, m5);
	m1 = _mm_subs_epi16(m1, m3);
	_mm_store_si128((__m128i*) bw, m1);

	/* L-values */
	m9 = _mm_set_epi8(LV_BW_SHUFFLE_MASK0);
	m10 = _mm_set_epi8(LV_BW_SHUFFLE_MASK1);
	m9 = _mm_shuffle_epi8(m0, m9);
	m10 = _mm_shuffle_epi8(m0, m10);

	m11 = _mm_adds_epi16(m6, m13);
	m12 = _mm_subs_epi16(m6, m13);
	m11 = _mm_adds_epi16(m11, m9);
	m12 = _mm_adds_epi16(m12, m10);

	/* Maximums */
#if defined(HAVE_SSE4_1) || defined(HAVE_SSE41)
	m13 = _mm_set1_epi16(32767);
	m14 = _mm_sub_epi16(m13, m11);
	m15 = _mm_sub_epi16(m13, m12);
	m14 = _mm_minpos_epu16(m14);
	m15 = _mm_minpos_epu16(m15);
	m14 = _mm_sub_epi16(m13, m14);
	m15 = _mm_sub_epi16(m13, m15);
#else
	MAXPOS(m11, m13, m14);
	MAXPOS(m12, m13, m15);
#endif
	m13 = _mm_sub_epi16(m15, m14);

	/* Return cast should truncate upper 16-bits */
	return _mm_cvtsi128_si32(m13);
}

#else

typedef union  __data128i {
	int16_t     m128i_i16[8];
	uint16_t    m128i_u16[8];
} __data128i;

void ci_set1_epi16(int16_t* dest, int16_t src);
void ci_setzero_si128(int16_t* dest);

void ci_set_epi16(int16_t* dest, short _W7, short _W6, short _W5, short _W4,
	short _W3, short _W2, short _W1, short _W0);

void ci_sign_epi16(int16_t* dest, int16_t* a, int16_t* b);

void ci_srai_epi16(int16_t* dest, int16_t* a, int imm8);

void ci_unpacklo_epi16(int16_t* dest, int16_t* a, int16_t* b);
void ci_unpackhi_epi16(int16_t* dest, int16_t* a, int16_t* b);

void ci_store_si128(int16_t* dest, int16_t* a);
void ci_load_si128(int16_t* dest, int16_t* src);

void ci_set_epi8(int8_t* dest, char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0);

void ci_adds_epi16(int16_t* dest, int16_t* a, int16_t* b);
void ci_subs_epi16(int16_t* dest, int16_t* a, int16_t* b);


void ci_shuffle_epi8(int8_t* dest, int8_t* a, int8_t* b);

void ci_max_epi16(int16_t* dest, int16_t* a, int16_t* b);

void ci_broadcastw_epi16(int16_t* dest, int16_t* a);

void ci_sub_epi16(int16_t* dest, int16_t* a, int16_t* b);

void ci_minpos_epu16(int16_t* dest, int16_t* a);

void ci_shuffle_epi32(int32_t* dst, int32_t* src, int d, int c, int b, int a);
void ci_shufflelo_epi16(int16_t* dst, int16_t* src, int d, int c, int b, int a);
void ci_shufflehi_epi16(int16_t* dst, int16_t* src, int d, int c, int b, int a);



#define MAXPOS_CI(M0,M1,M2) \
{ \
	ci_shuffle_epi32((int32_t*)M1, (int32_t*)M0, 0, 0, 3, 2); \
	ci_max_epi16(M2, M0, M1); \
	ci_shufflelo_epi16(M1, M2, 0, 0, 3, 2); \
	ci_max_epi16(M2, M2, M1); \
	ci_shufflelo_epi16(M1, M2, 0, 0, 0, 1); \
	ci_max_epi16(M2, M2, M1); \
}



static inline int16_t gen_fw_metrics(int16_t* bm, int8_t x, int8_t z,
	int16_t* sums_p, int16_t* sums_c, int16_t le)
{
	__data128i m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13;

	ci_set1_epi16(m0.m128i_i16, x);
	ci_set1_epi16(m1.m128i_i16, z);
	ci_set1_epi16(m2.m128i_i16, le);
	ci_setzero_si128(m3.m128i_i16);

	ci_set_epi16(m4.m128i_i16, LTE_SYSTEM_FW_SHUFFLE);
	ci_set_epi16(m5.m128i_i16, LTE_PARITY_FW_SHUFFLE);

	/* Branch metrics */
	ci_sign_epi16(m6.m128i_i16, m0.m128i_i16, m4.m128i_i16);
	ci_sign_epi16(m7.m128i_i16, m1.m128i_i16, m5.m128i_i16);
	ci_sign_epi16(m8.m128i_i16, m2.m128i_i16, m4.m128i_i16);
	ci_srai_epi16(m8.m128i_i16, m8.m128i_i16, 1);

	ci_adds_epi16(m6.m128i_i16, m6.m128i_i16, m7.m128i_i16);
	ci_adds_epi16(m6.m128i_i16, m6.m128i_i16, m8.m128i_i16);
	ci_subs_epi16(m7.m128i_i16, m3.m128i_i16, m6.m128i_i16);


	/* Pre-interleave for backward recursion */
	ci_unpacklo_epi16(m8.m128i_i16, m6.m128i_i16, m7.m128i_i16);
	ci_store_si128(bm, m8.m128i_i16);

	/* Forward metrics */
	ci_load_si128(m9.m128i_i16, sums_p);
	ci_set_epi8((int8_t*)m10.m128i_i16, FW_SHUFFLE_MASK0);
	ci_set_epi8((int8_t*)m11.m128i_i16, FW_SHUFFLE_MASK1);
	//
	ci_shuffle_epi8((int8_t*)m12.m128i_i16, (int8_t*)m9.m128i_i16, (int8_t*)m10.m128i_i16);
	ci_shuffle_epi8((int8_t*)m13.m128i_i16, (int8_t*)m9.m128i_i16, (int8_t*)m11.m128i_i16);
	ci_adds_epi16(m12.m128i_i16, m12.m128i_i16, m6.m128i_i16);
	ci_adds_epi16(m13.m128i_i16, m13.m128i_i16, m7.m128i_i16);
	ci_max_epi16(m0.m128i_i16, m12.m128i_i16, m13.m128i_i16);

	ci_broadcastw_epi16(m1.m128i_i16, m0.m128i_i16);
	ci_subs_epi16(m0.m128i_i16, m0.m128i_i16, m1.m128i_i16);

	ci_store_si128((int16_t*)sums_c, m0.m128i_i16);

	///* Return 32-bit integer with garbage in upper 16-bits */
	//return _mm_cvtsi128_si32(m1);
	int32_t* res_data = (int32_t*)m1.m128i_i16;
	return *res_data;
}

static inline int16_t gen_bw_metrics(int16_t* bm, const int8_t z,
	int16_t* fw, int16_t* bw, int16_t norm)
{
	__data128i m0, m1, m3, m4, m5, m6, m9, m10, m11, m12, m13, m14, m15;

	ci_set1_epi16(m0.m128i_i16, z);
	ci_set_epi16(m1.m128i_i16, LTE_PARITY_BW_SHUFFLE);
	//
	//	/* Partial branch metrics */
	ci_sign_epi16(m13.m128i_i16, m0.m128i_i16, m1.m128i_i16);
	//
	//	/* Backward metrics */
	ci_load_si128(m0.m128i_i16, bw);
	ci_load_si128(m1.m128i_i16, bm);
	ci_load_si128(m6.m128i_i16, fw);
	ci_set1_epi16(m3.m128i_i16, norm);
	//
	ci_unpacklo_epi16(m4.m128i_i16, m0.m128i_i16, m0.m128i_i16);
	ci_unpackhi_epi16(m5.m128i_i16, m0.m128i_i16, m0.m128i_i16);
	ci_adds_epi16(m4.m128i_i16, m4.m128i_i16, m1.m128i_i16);
	ci_subs_epi16(m5.m128i_i16, m5.m128i_i16, m1.m128i_i16);
	//
	ci_max_epi16(m1.m128i_i16, m4.m128i_i16, m5.m128i_i16);
	ci_subs_epi16(m1.m128i_i16, m1.m128i_i16, m3.m128i_i16);
	ci_store_si128(bw, m1.m128i_i16);
	//
	//	/* L-values */
	ci_set_epi8((int8_t*)m9.m128i_i16, LV_BW_SHUFFLE_MASK0);
	ci_set_epi8((int8_t*)m10.m128i_i16, LV_BW_SHUFFLE_MASK1);
	ci_shuffle_epi8((int8_t*)m9.m128i_i16, (int8_t*)m0.m128i_i16, (int8_t*)m9.m128i_i16);
	ci_shuffle_epi8((int8_t*)m10.m128i_i16, (int8_t*)m0.m128i_i16, (int8_t*)m10.m128i_i16);
	//
	ci_adds_epi16(m11.m128i_i16, m6.m128i_i16, m13.m128i_i16);
	ci_subs_epi16(m12.m128i_i16, m6.m128i_i16, m13.m128i_i16);
	ci_adds_epi16(m11.m128i_i16, m11.m128i_i16, m9.m128i_i16);
	ci_adds_epi16(m12.m128i_i16, m12.m128i_i16, m10.m128i_i16);
	//
	//	/* Maximums */

	MAXPOS_CI(m11.m128i_i16, m13.m128i_i16, m14.m128i_i16);
	MAXPOS_CI(m12.m128i_i16, m13.m128i_i16, m15.m128i_i16);

	ci_sub_epi16(m13.m128i_i16, m15.m128i_i16, m14.m128i_i16);
	//
	//	/* Return cast should truncate upper 16-bits */
	int32_t* res_data = (int32_t*)m13.m128i_i16;
	return *res_data;
}



void ci_set1_epi16(int16_t* dest, int16_t src)
{
	for (size_t i = 0; i < 8; i++)
	{
		dest[i] = src;
	}
}

void ci_setzero_si128(int16_t* dest)
{
	for (size_t i = 0; i < 8; i++)
	{
		dest[i] = 0;
	}
}

void ci_set_epi16(int16_t* dest, short _W7, short _W6, short _W5, short _W4,
	short _W3, short _W2, short _W1, short _W0)
{
	dest[0] = _W0;
	dest[1] = _W1;
	dest[2] = _W2;
	dest[3] = _W3;
	dest[4] = _W4;
	dest[5] = _W5;
	dest[6] = _W6;
	dest[7] = _W7;
}


void ci_sign_epi16(int16_t* dest, int16_t* a, int16_t* b)
{
	for (size_t i = 0; i < 8; i++)
	{
		if (b[i] < 0)
		{
			dest[i] = -a[i];
		}
		else if (b[i] == 0)
		{
			dest[i] = 0;
		}
		else
		{
			dest[i] = a[i];
		}
	}
}

void ci_srai_epi16(int16_t* dest, int16_t* a, int imm8)
{
	for (size_t i = 0; i < 8; i++)
	{
		if (imm8 > 15)
		{
			dest[i] = (a[i] ? 0xFFFF : 0x0);
		}
		else
		{
			dest[i] = (a[i] >> imm8);
		}
	}
}

void ci_adds_epi16(int16_t* dest, int16_t* a, int16_t* b)
{
	for (size_t i = 0; i < 8; i++)
	{
		dest[i] = a[i] + b[i];
	}
}

void ci_subs_epi16(int16_t* dest, int16_t* a, int16_t* b)
{
	for (size_t i = 0; i < 8; i++)
	{
		dest[i] = a[i] - b[i];
	}
}


void ci_unpacklo_epi16(int16_t* dest, int16_t* a, int16_t* b)
{
	dest[0] = a[0];
	dest[1] = b[0];
	dest[2] = a[1];
	dest[3] = b[1];
	dest[4] = a[2];
	dest[5] = b[2];
	dest[6] = a[3];
	dest[7] = b[3];
}

void ci_unpackhi_epi16(int16_t* dest, int16_t* a, int16_t* b)
{
	dest[0] = a[4];
	dest[1] = b[4];
	dest[2] = a[5];
	dest[3] = b[5];
	dest[4] = a[6];
	dest[5] = b[6];
	dest[6] = a[7];
	dest[7] = b[7];
}

void ci_store_si128(int16_t* dest, int16_t* a)
{
	for (size_t i = 0; i < 8; i++)
	{
		dest[i] = a[i];
	}
}

void ci_load_si128(int16_t* dest, int16_t* src)
{
	for (size_t i = 0; i < 8; i++)
	{
		dest[i] = src[i];
	}
}

void ci_set_epi8(int8_t* dest, char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0)
{
	dest[15] = e15;
	dest[14] = e14;
	dest[13] = e13;
	dest[12] = e12;
	dest[11] = e11;
	dest[10] = e10;
	dest[9] = e9;
	dest[8] = e8;
	dest[7] = e7;
	dest[6] = e6;
	dest[5] = e5;
	dest[4] = e4;
	dest[3] = e3;
	dest[2] = e2;
	dest[1] = e1;
	dest[0] = e0;
}

void ci_shuffle_epi8(int8_t* dest, int8_t* a, int8_t* b)
{
	for (int i = 0; i < 16; i++)
	{
		dest[i] = (b[i] < 0) ? 0 : a[b[i] % 16];
	}
}


void ci_max_epi16(int16_t* dest, int16_t* a, int16_t* b)
{
	for (size_t i = 0; i < 8; i++)
	{
		if (a[i] > b[i]) dest[i] = a[i];
		else dest[i] = b[i];
	}
}

void ci_broadcastw_epi16(int16_t* dest, int16_t* a)
{
	for (size_t i = 0; i < 8; i++)
	{
		dest[i] = a[0];
	}
}


void ci_sub_epi16(int16_t* dest, int16_t* a, int16_t* b)
{
	for (size_t i = 0; i < 8; i++)
	{
		dest[i] = a[i] - b[i];
	}
}


void ci_minpos_epu16(int16_t* dest, int16_t* a)
{
	int index = 0;
	int min = a[0];
	for (size_t i = 0; i < 8; i++)
	{
		if (a[i] < min)
		{
			min = a[i];
			index = i;
		}
		dest[i] = 0;
	}
	dest[0] = min;
	dest[1] = index;
}

void ci_shuffle_epi32(int32_t* dst, int32_t* src, int d, int c, int b, int a)
{
	dst[0] = src[a];
	dst[1] = src[b];
	dst[2] = src[c];
	dst[3] = src[d];
}

void ci_shufflelo_epi16(int16_t* dst, int16_t* src, int d, int c, int b, int a)
{
	dst[0] = src[a];
	dst[1] = src[b];
	dst[2] = src[c];
	dst[3] = src[d];
	for (size_t i = 4; i < 8; i++)
	{
		dst[i] = src[i];
	}
}

void ci_shufflehi_epi16(int16_t* dst, int16_t* src, int d, int c, int b, int a)
{
	for (size_t i = 0; i < 4; i++)
	{
		dst[i] = src[i];
	}
	dst[4] = src[a + 4];
	dst[5] = src[b + 4];
	dst[6] = src[c + 4];
	dst[7] = src[d + 4];
}


#endif /* HAVE_SSE3 */
